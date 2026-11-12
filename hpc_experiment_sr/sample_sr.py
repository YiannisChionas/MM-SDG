"""Cascaded sampling: stage 1 (hpc_experiment's trained 64x64 checkpoint)
generates a low-res image per class, which is upsampled and fed as
conditioning (along with the class label again) into stage 2 (this
experiment's 224x224 model) for the final high-res output.

Matches the paper's two-stage inference (low-res DDPM -> upsample -> super-
resolution DDPM). Resumable: counts existing .png files per class and only
generates the remainder.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler

from data_utils_sr import CANCER_TYPES, SRConditionedUnet, Stage1ClassConditionedUnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=str, required=True,
                         help="hpc_experiment's trained checkpoint.pt (64x64 class-conditioned model)")
    parser.add_argument("--stage2-checkpoint", type=str, required=True,
                         help="This experiment's trained checkpoint.pt (224x224 SR model)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--per-class", type=int, default=100,
                         help="Synthetic images per cancer-type class (default 100 -> 1000 total)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--low-res-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    num_classes = len(CANCER_TYPES)

    stage1 = Stage1ClassConditionedUnet(num_classes=num_classes, image_size=args.low_res_size).to(device)
    ckpt1 = torch.load(args.stage1_checkpoint, map_location=device)
    stage1.load_state_dict(ckpt1["model_state_dict"])
    stage1.eval()
    print(f"Loaded stage-1 checkpoint from epoch {ckpt1['epoch']}", flush=True)

    stage2 = SRConditionedUnet(num_classes=num_classes, image_size=args.image_size).to(device)
    ckpt2 = torch.load(args.stage2_checkpoint, map_location=device)
    stage2.load_state_dict(ckpt2["model_state_dict"])
    stage2.eval()
    print(f"Loaded stage-2 checkpoint from epoch {ckpt2['epoch']}", flush=True)

    scheduler1 = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scheduler2 = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, cancer_type in enumerate(CANCER_TYPES):
        class_dir = out_dir / cancer_type
        class_dir.mkdir(parents=True, exist_ok=True)
        n_done = len(list(class_dir.glob("*.png")))
        n_remaining = args.per_class - n_done
        if n_remaining <= 0:
            print(f"{cancer_type}: already has {n_done} images, skipping", flush=True)
            continue
        print(f"{cancer_type} (class {class_idx}): generating {n_remaining} images "
              f"({n_done} already done)", flush=True)

        generated = 0
        while generated < n_remaining:
            bs = min(args.batch_size, n_remaining - generated)
            y = torch.full((bs,), class_idx, dtype=torch.long, device=device)

            # Stage 1: generate low-res (64x64) image
            with torch.no_grad():
                x_low = torch.randn(bs, 3, args.low_res_size, args.low_res_size, device=device)
                for t in scheduler1.timesteps:
                    residual = stage1(x_low, t, y)
                    x_low = scheduler1.step(residual, t, x_low).prev_sample

                # Upsample stage-1 output to feed as stage-2 conditioning
                low_res_cond = F.interpolate(
                    x_low, size=(args.image_size, args.image_size),
                    mode="bilinear", align_corners=False,
                )

                # Stage 2: refine to high-res (224x224), conditioned on upsampled low-res + class
                x_high = torch.randn(bs, 3, args.image_size, args.image_size, device=device)
                for t in scheduler2.timesteps:
                    residual = stage2(x_high, low_res_cond, t, y)
                    x_high = scheduler2.step(residual, t, x_high).prev_sample

            imgs = ((x_high.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(bs):
                idx = n_done + generated + i
                Image.fromarray(imgs[i]).save(class_dir / f"{idx:05d}.png")
            generated += bs
            print(f"  {cancer_type}: {n_done + generated}/{args.per_class}", flush=True)

    print("Cascaded sampling complete.", flush=True)


if __name__ == "__main__":
    main()
