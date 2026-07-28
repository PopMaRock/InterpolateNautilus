#!/usr/bin/env python3
"""
InterpolateNautilus — Unified frame interpolation CLI.

Supports multiple VFI models selectable via --model:
  rife      Base RIFE (v4.x architecture)
  sg-rife   SG-RIFE with DINOv3 semantic injection
"""

import os
import sys
import cv2
import torch
import argparse
import warnings

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# SageAttention patching
# =============================================================================

def _patch_sdpa_with_sage():
    """Patch torch.nn.functional.scaled_dot_product_attention with sageattn.

    The sageattn API is a near drop-in for F.scaled_dot_product_attention
    when using the default (B, head_num, seq_len, head_dim) layout ("HND").
    We wrap it to ignore attn_mask/dropout_p/scale (not supported by sageattn).
    """
    try:
        from sageattention import sageattn
    except ImportError:
        print("Warning: sageattention not installed. Falling back to default SDPA.")
        return False

    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None):
        # sageattn requires FP16/BF16 inputs; auto-cast if needed
        orig_dtype = query.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            query_fp = query.half()
            key_fp = key.half()
            value_fp = value.half()
        else:
            query_fp = query
            key_fp = key
            value_fp = value
        out = sageattn(query_fp, key_fp, value_fp, tensor_layout="HND",
                       is_causal=is_causal)
        return out.to(orig_dtype)

    torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
    print("SageAttention: patched F.scaled_dot_product_attention with sageattn")
    return True

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# =============================================================================
# Audio transfer helper
# =============================================================================


def transferAudio(sourceVideo, targetVideo):
    import shutil
    import subprocess

    tempAudio = "./temp/audio.mkv"

    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")

    subprocess.run(
        ["ffmpeg", "-y", "-i", sourceVideo, "-c:a", "copy", "-vn", tempAudio],
        capture_output=True,
    )

    targetNoAudio = (
        os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    )
    os.rename(targetVideo, targetNoAudio)

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", targetNoAudio, "-i", tempAudio, "-c", "copy", targetVideo],
        capture_output=True,
    )

    if os.path.getsize(targetVideo) == 0:
        tempAudio = "./temp/audio.m4a"
        subprocess.run(
            ["ffmpeg", "-y", "-i", sourceVideo, "-c:a", "aac", "-b:a", "160k", "-vn", tempAudio],
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", targetNoAudio, "-i", tempAudio, "-c", "copy", targetVideo],
            capture_output=True,
        )
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    shutil.rmtree("temp")


# =============================================================================
# Parse arguments
# =============================================================================

parser = argparse.ArgumentParser(description="InterpolateNautilus — Multi-VFI frame interpolation")
parser.add_argument("--model", dest="model", type=str, default="rife",
                    choices=["rife", "sg-rife"],
                    help="VFI model to use")
parser.add_argument("--img", dest="img", nargs=2, default=None,
                    help="Two input image paths for image interpolation")
parser.add_argument("--video", dest="video", type=str, default=None,
                    help="Input video file for video interpolation")
parser.add_argument("--output", dest="output", type=str, default=None,
                    help="Output video path (video mode only)")
parser.add_argument("--exp", default=4, type=int,
                    help="Interpolation factor: produces 2^exp frames per pair")
parser.add_argument("--ratio", default=0, type=float,
                    help="Single-frame ratio between 0.0 and 1.0 (image mode)")
parser.add_argument("--rthreshold", default=0.02, type=float,
                    help="Ratio tolerance for bisection search")
parser.add_argument("--rmaxcycles", default=8, type=int,
                    help="Max bisection cycles")
parser.add_argument("--fps", dest="fps", type=int, default=None,
                    help="Output FPS (video mode)")
parser.add_argument("--scale", dest="scale", type=float, default=1.0,
                    help="Resolution scale (0.25, 0.5, 1.0, 2.0, 4.0)")
parser.add_argument("--fp16", dest="fp16", action="store_true",
                    help="Half-precision inference")
parser.add_argument("--UHD", dest="UHD", action="store_true",
                    help="4K mode (auto scale=0.5)")
parser.add_argument("--png", dest="png", action="store_true",
                    help="Output PNG sequence instead of video")
parser.add_argument("--ext", dest="ext", type=str, default="mp4",
                    help="Output video container format")
parser.add_argument("--montage", dest="montage", action="store_true",
                    help="Side-by-side montage mode (video)")
parser.add_argument("--with-sage", dest="use_sage", action="store_true",
                    help="Use SageAttention for accelerated attention (SG-RIFE only)")
parser.add_argument("--model-dir", dest="modelDir", type=str, default=None,
                    help="Override checkpoint directory")

args = parser.parse_args()

if args.img is None and args.video is None:
    parser.error("Either --img or --video must be specified")

# =============================================================================
# Device setup
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

# =============================================================================
# Model loading
# =============================================================================

print(f"Loading model: {args.model}")

if args.use_sage and args.model == "rife":
    print("Note: --with-sage has no effect on base RIFE (no attention layers in the architecture)")

if args.model == "sg-rife":
    if args.use_sage:
        _patch_sdpa_with_sage()

    from vfi_models.sg_rife.sg_rife_arch import Model as SGRifeModel
    from vfi_models.sg_rife.dino_config import DinoConfig

    checkpoint_dir = args.modelDir or os.path.join(_SCRIPT_DIR, "checkpoints", "sg_rife")
    model = SGRifeModel()
    model.load_model(os.path.join(checkpoint_dir, "flownet.pkl"))
    print("Loaded SG-RIFE (DINO-enhanced) model")

elif args.model == "rife":
    from vfi_models.rife.rife_arch import RifeModel

    checkpoint_dir = args.modelDir or os.path.join(_SCRIPT_DIR, "checkpoints", "rife")
    model = RifeModel()
    model.load_model(os.path.join(checkpoint_dir, "rife49.pth"))
    print("Loaded RIFE v4.x model")

model.eval()
model.device()

# =============================================================================
# Image interpolation
# =============================================================================

if args.img is not None:
    import torch.nn.functional as F

    if args.img[0].endswith(".exr") and args.img[1].endswith(".exr"):
        img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)
    else:
        img0 = cv2.imread(args.img[0], cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(args.img[1], cv2.IMREAD_UNCHANGED)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    if args.ratio:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if args.ratio <= img0_ratio + args.rthreshold / 2:
            middle = img0
        elif args.ratio >= img1_ratio - args.rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(args.rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = (img0_ratio + img1_ratio) / 2
                lower = args.ratio - (args.rthreshold / 2)
                upper = args.ratio + (args.rthreshold / 2)
                if lower <= middle_ratio <= upper:
                    break
                if args.ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(args.exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp

    if not os.path.exists("output"):
        os.mkdir("output")
    for i in range(len(img_list)):
        if args.img[0].endswith(".exr") and args.img[1].endswith(".exr"):
            out = (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w]
            cv2.imwrite(
                f"output/img{i}.exr", out,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF],
            )
        else:
            out = (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            cv2.imwrite(f"output/img{i}.png", out)
    print(f"Saved {len(img_list)} frames to output/")
    sys.exit(0)

# =============================================================================
# Video interpolation
# =============================================================================

import numpy as np
import _thread
from queue import Queue
from tqdm import tqdm
from pytorch_msssim import ssim_matlab

if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

videoCapture = cv2.VideoCapture(args.video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
tot_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
if args.fps is None:
    fpsNotAssigned = True
    args.fps = fps * (2 ** args.exp)
else:
    fpsNotAssigned = False

success, first_frame = videoCapture.read()
if not success:
    raise RuntimeError(f"Failed to read video: {args.video}")
lastframe = first_frame.copy()

fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
video_path_wo_ext, ext = os.path.splitext(args.video)
print(f"{video_path_wo_ext}.{args.ext}, {tot_frame} frames in total, {fps}FPS to {args.fps}FPS")

if args.png == False and fpsNotAssigned == True:
    print("The audio will be merged after interpolation process")
else:
    print("Will not merge audio because using png or fps flag!")

h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists("vid_out"):
        os.mkdir("vid_out")
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = f"{video_path_wo_ext}_{2 ** args.exp}X_{int(np.round(args.fps))}fps.{args.ext}"
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h), isColor=True)


def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite(f"vid_out/{cnt:0>7d}.png", item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item)


def build_read_buffer(user_args, read_buffer, videogen):
    try:
        while True:
            success, frame = videogen.read()
            if not success:
                break
            frame = frame.copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def make_inference(I0, I1, n):
    global model
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n // 2)
    second_half = make_inference(middle, I1, n=n // 2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def pad_image(img):
    if args.fp16:
        return torch.nn.functional.pad(img, padding).half()
    else:
        return torch.nn.functional.pad(img, padding)


if args.montage:
    left = w // 4
    w = w // 2
tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videoCapture))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = (
    torch.from_numpy(np.transpose(lastframe, (2, 0, 1)))
    .to(device, non_blocking=True)
    .unsqueeze(0)
    .float()
    / 255.0
)
I1 = pad_image(I1)
temp = None

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = (
        torch.from_numpy(np.transpose(frame, (2, 0, 1)))
        .to(device, non_blocking=True)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    I1 = pad_image(I1)
    I0_small = torch.nn.functional.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
    I1_small = torch.nn.functional.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get()
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = (
            torch.from_numpy(np.transpose(frame, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, args.scale)
        I1_small = torch.nn.functional.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    if ssim < 0.2:
        output = []
        for i in range((2 ** args.exp) - 1):
            output.append(I0)
    else:
        output = make_inference(I0, I1, 2 ** args.exp - 1) if args.exp else []

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
            write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)

write_buffer.put(None)

import time
while not write_buffer.empty():
    time.sleep(0.1)
pbar.close()
if vid_out is not None:
    vid_out.release()

# Merge audio
if args.png == False and fpsNotAssigned == True and args.video is not None:
    try:
        transferAudio(args.video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = (
            os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        )
        if os.path.exists(targetNoAudio):
            os.rename(targetNoAudio, vid_out_name)

print("Done.")