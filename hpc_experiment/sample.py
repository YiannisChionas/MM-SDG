"""Generate synthetic HEST patches from a trained checkpoint, per cancer type.

Resumable: counts existing .png files per class and only generates the
remainder, so a job that hits the time limit mid-run can simply be
resubmitted (same --output-dir) to pick up where it left off.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import DDPMScheduler

from data_utils import CANCER_TYPES, ClassConditionedUnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--per-class", type=int, default=1000,
                         help="Synthetic images to generate per cancer-type class (default 1000 -> 10,000 total)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    model = ClassConditionedUnet(num_classes=len(CANCER_TYPES), image_size=args.image_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}", flush=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

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
            x = torch.randn(bs, 3, args.image_size, args.image_size, device=device)
            y = torch.full((bs,), class_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                for t in noise_scheduler.timesteps:
                    residual = model(x, t, y)
                    x = noise_scheduler.step(residual, t, x).prev_sample

            imgs = ((x.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(bs):
                idx = n_done + generated + i
                Image.fromarray(imgs[i]).save(class_dir / f"{idx:05d}.png")
            generated += bs
            print(f"  {cancer_type}: {n_done + generated}/{args.per_class}", flush=True)

    print("Sampling complete.", flush=True)


if __name__ == "__main__":
    main()
