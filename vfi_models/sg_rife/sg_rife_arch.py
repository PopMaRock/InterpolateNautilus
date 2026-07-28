# SG-RIFE: Single-file architecture (warplayer + IFNet_dino + refine_dino + Model wrapper)
# Based on RIFE (https://github.com/hzwer/ECCV2022-RIFE) and SG-RIFE (arXiv:2512.18241)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dino_config import DinoConfig
from .dino_modules import DinoWrapper, FAPM_Encoder, FAPM_Refiner, DinoFusion

# =============================================================================
# Backward warping utility
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear',
                                           padding_mode='border', align_corners=True)


# =============================================================================
# Shared building blocks
# =============================================================================

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes),
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes),
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def get_downsampled_flow(flow, target_h, target_w):
    B, C, H, W = flow.shape
    flow_low = F.interpolate(flow, size=(target_h, target_w), mode="area")
    scale_w = target_w / W
    scale_h = target_h / H
    flow_low_scaled = torch.cat(
        [
            flow_low[:, 0:1] * scale_w,
            flow_low[:, 1:2] * scale_h,
            flow_low[:, 2:3] * scale_w,
            flow_low[:, 3:4] * scale_h,
        ],
        dim=1,
    )
    return flow_low_scaled


# =============================================================================
# IFBlock
# =============================================================================

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False,
            )
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear",
                            align_corners=False)
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(
            tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


# =============================================================================
# Contextnet
# =============================================================================

c = 16


class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = (
            F.interpolate(
                flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                recompute_scale_factor=False,
            )
            * 0.5
        )
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = (
            F.interpolate(
                flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                recompute_scale_factor=False,
            )
            * 0.5
        )
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = (
            F.interpolate(
                flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                recompute_scale_factor=False,
            )
            * 0.5
        )
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = (
            F.interpolate(
                flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                recompute_scale_factor=False,
            )
            * 0.5
        )
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


# =============================================================================
# Unet (DINO-enhanced)
# =============================================================================

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        pixel_input_dim = 17  # 3+3+3+3+1+4

        self.down0 = Conv2(pixel_input_dim, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)

        self.fusion_s3 = DinoFusion(16 * c, 8)  # 256 channels
        self.adapt_s2 = nn.ConvTranspose2d(
            8 * c, 8 * c, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.fusion_s2 = DinoFusion(8 * c, 4)  # 128 channels

        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1,
                mask, flow, c0, c1, dino_feats0, dino_feats1):

        pixel_feats = torch.cat(
            (img0, img1, warped_img0, warped_img1, mask, flow), 1
        )
        s0 = self.down0(pixel_feats)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))

        adapted_d0_feat = self.adapt_s2(dino_feats0[0])
        adapted_d1_feat = self.adapt_s2(dino_feats1[0])
        s2, offsets2 = self.fusion_s2(s2, (adapted_d0_feat, adapted_d1_feat))

        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        s3, offsets3 = self.fusion_s3(s3, (dino_feats0[1], dino_feats1[1]))

        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x), offsets2 + offsets3


# =============================================================================
# IFNet (with DINO fusion)
# =============================================================================

class IFNet(nn.Module):
    def __init__(self, dino_in_channels, dino_patch_size):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        self.contextnet = Contextnet()

        self.cfg = DinoConfig()
        self.dino_embed_dim = dino_in_channels
        self.dino_patch_size = dino_patch_size
        self.dino_compressor = FAPM_Encoder(
            in_dim=dino_in_channels,
            rank=self.cfg.compressor_rank,
            num_layers=2,
        )
        self.dino_refiner = FAPM_Refiner(
            rank=self.cfg.compressor_rank, out_ch_list=[128, 256]
        )
        self.unet = Unet()

    def forward(self, x, dino_feats, scale_list=None, timestep=0.5):
        if scale_list is None:
            scale_list = [4, 2, 1]
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is None:
                flow, mask = stu[i](
                    torch.cat((img0, img1), 1), None, scale=scale_list[i]
                )
            else:
                cat = torch.cat(
                    (img0, img1, warped_img0, warped_img1, mask), 1
                )
                flow_d, mask_d = stu[i](cat, flow, scale=scale_list[i])
                flow = flow + flow_d
                mask = mask + mask_d
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            cat = torch.cat(
                (img0, img1, warped_img0, warped_img1, mask, gt), 1
            )
            flow_d, mask_d = self.block_tea(cat, flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = (
                warped_img0_teacher * mask_teacher
                + warped_img1_teacher * (1 - mask_teacher)
            )
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (
                1 - mask_list[i]
            )
            if gt.shape[1] == 3:
                student_error = (merged[i] - gt).abs().mean(1, True)
                teacher_error = (merged_teacher - gt).abs().mean(1, True)
                loss_mask = (
                    (student_error > (teacher_error + 0.01)).float().detach()
                )
                flow_diff = flow_teacher.detach() - flow_list[i]
                flow_dist = (flow_diff ** 2).mean(1, True) ** 0.5
                loss_distill += (flow_dist * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        final_flow = flow_list[2]  # [B, 4, H, W]

        d0_list, d1_list = dino_feats
        B, C_d, H_d, W_d = dino_feats[0][0].shape

        # List of [2B, C, H_d, W_d]
        combined_dino_inputs = []
        for feat0, feat1 in zip(d0_list, d1_list):
            combined_dino_inputs.append(torch.cat([feat0, feat1], dim=0))

        # List of [2B, C_comp, H_d, W_d]
        compressed_combined = self.dino_compressor(combined_dino_inputs)

        # [B, 4, H_d, W_d]
        flow_down = get_downsampled_flow(final_flow, H_d, W_d)

        # [2B, 2, H_d, W_d]
        flow_combined = torch.cat([flow_down[:, :2], flow_down[:, 2:4]], dim=0)

        warped_combined = []
        for feat in compressed_combined:
            warped_combined.append(warp(feat, flow_combined))

        refined_combined = self.dino_refiner(warped_combined)

        dino_finals0 = []
        dino_finals1 = []
        B = x.shape[0]

        for r in refined_combined:
            r0, r1 = torch.split(r, B, dim=0)
            dino_finals0.append(r0)
            dino_finals1.append(r1)

        tmp, offsets = self.unet(
            img0, img1, warped_img0, warped_img1,
            mask, flow, c0, c1, dino_finals0, dino_finals1,
        )
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return (
            flow_list,
            mask_list[2],
            merged,
            flow_teacher,
            merged_teacher,
            loss_distill,
            offsets,
        )


# =============================================================================
# Model (inference-only wrapper)
# =============================================================================

class Model:
    def __init__(self, dino_cfg=None):
        if dino_cfg is None:
            dino_cfg = DinoConfig()
        self.dino_cfg = dino_cfg
        self.dino = DinoWrapper(self.dino_cfg)
        self.flownet = IFNet(
            dino_in_channels=self.dino.embed_dim,
            dino_patch_size=self.dino.patch_size,
        )
        self.device()

    def eval(self):
        self.flownet.eval()
        self.dino.eval()

    def device(self):
        self.flownet.to(device)
        self.dino.to(device)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        self.flownet.load_state_dict(new_state_dict)

    def inference(self, img0, img1, scale=1, scale_list=None, timestep=0.5):
        if scale_list is None:
            scale_list = [4, 2, 1]
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        feats0 = self.dino.get_features(img0, use_grad=False)
        feats1 = self.dino.get_features(img1, use_grad=False)
        dino_feats = (feats0, feats1)
        _, _, merged, _, _, _, _ = self.flownet(
            imgs, dino_feats, scale_list, timestep=timestep
        )
        return merged[2]