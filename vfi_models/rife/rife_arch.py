# Base RIFE 4.7/4.10 architecture — single-file port from ComfyUI frame interpolation

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


# =============================================================================
# Backward warping
# =============================================================================

def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear',
                         padding_mode='border', align_corners=True)


# =============================================================================
# Building blocks
# =============================================================================

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.2, True),
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


# =============================================================================
# Encode head (4.7+)
# =============================================================================

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


# =============================================================================
# IFBlock (4.7/4.10 style with ResConv + PixelShuffle)
# =============================================================================

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, lastconv_scale=6):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * lastconv_scale, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1.0):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear",
                            align_corners=False)
        if flow is not None:
            flow = (F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear",
                                align_corners=False) * 1.0 / scale)
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow_out = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow_out, mask


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
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


# =============================================================================
# Unet
# =============================================================================

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        c = 16
        self.down0 = Conv2(17, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


# =============================================================================
# IFNet (4.7-style with encode head)
# =============================================================================

class IFNet47(nn.Module):
    def __init__(self):
        super(IFNet47, self).__init__()
        self.block0 = IFBlock(15, c=192, lastconv_scale=6)
        self.block1 = IFBlock(20, c=128, lastconv_scale=6)
        self.block2 = IFBlock(20, c=96,  lastconv_scale=6)
        self.block3 = IFBlock(20, c=64,  lastconv_scale=6)
        self.encode = Head()
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, img0, img1, timestep=0.5):
        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0_p = F.pad(img0, padding)
        img1_p = F.pad(img1, padding)

        if not torch.is_tensor(timestep):
            timestep = (img0_p[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0_p.shape[2], img0_p.shape[3])

        f0 = self.encode(img0_p[:, :3])
        f1 = self.encode(img1_p[:, :3])

        scale_list = [16, 8, 4, 2, 1]
        warped_img0 = img0_p
        warped_img1 = img1_p
        flow = None
        mask = None
        blocks = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](
                    torch.cat((img0_p[:, :3], img1_p[:, :3], f0, f1, timestep), 1),
                    None, scale=scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                head_input = [warped_img0[:, :3], warped_img1[:, :3], wf0, wf1,
                              timestep, mask]
                fd, m0 = blocks[i](
                    torch.cat(head_input, 1),
                    flow, scale=scale_list[i],
                )
                flow = flow + fd
                mask = m0
            warped_img0 = warp(img0_p, flow[:, :2])
            warped_img1 = warp(img1_p, flow[:, 2:4])

        mask_merged = torch.sigmoid(mask)
        merged_final = warped_img0 * mask_merged + warped_img1 * (1 - mask_merged)

        c0 = self.contextnet(img0_p, flow[:, :2])
        c1 = self.contextnet(img1_p, flow[:, 2:4])
        tmp = self.unet(img0_p, img1_p, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged_final = torch.clamp(merged_final + res, 0, 1)

        return merged_final[:, :, :h, :w]


# =============================================================================
# Inference wrapper
# =============================================================================

class RifeModel:
    def __init__(self):
        self.flownet = IFNet47()
        self.device()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        missing, unexpected = self.flownet.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Note: {len(missing)} model keys not in checkpoint (random init)")
        if unexpected:
            print(f"Note: {len(unexpected)} checkpoint keys not in model (ignored)")

    def inference(self, img0, img1, scale=1.0):
        return self.flownet(img0, img1)