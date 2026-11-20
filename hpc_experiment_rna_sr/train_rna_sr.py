"""Train the RNA-conditioned cascaded super-resolution (stage 2) model:
conditions on a degraded low-res version of each real patch + the frozen
VAE's embedding of that spot's real gene expression, learns to recover the
native 224x224 image. CCRCC train-split WSIs only (official HEST-bench
fold, same as hpc_experiment_rna).

Uses accelerate (same pattern as hpc_experiment_sr/train_sr.py) and syncs
checkpoint.pt to --persist-dir after EVERY epoch (lesson learned from that
experiment: a SLURM TIMEOUT kill skips the job script's trailing cp -a,
silently losing an already-completed epoch).
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from accelerate import Accelerator

from data_utils_rna_sr import (
    CANCER_TYPE, load_gene_panel, load_split_sample_ids, build_patch_expr_index,
    GeneExpressionNormalizer, HestPatchExprPairDataset, BetaVAE, RnaSRConditionedUnet,
)


def save_checkpoint(path, model, optimizer, epoch, losses):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }, path)


def load_frozen_vae(vae_ckpt_path, device):
    ckpt = torch.load(vae_ckpt_path, map_location=device)
    vae = BetaVAE(input_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"], latent_dim=ckpt["latent_dim"])
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()
    normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])
    return vae, normalizer, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10,
                         help="Cumulative epoch target for this invocation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--low-res-size", type=int, default=64)
    parser.add_argument("--cond-aug-max-std", type=float, default=0.2,
                         help="Max std of Gaussian noise added to the low-res conditioning during "
                              "training, to bridge the train (clean degraded real image) vs sampling "
                              "(stage-1's actual imperfect DDPM output) distribution gap. 0 disables it.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--persist-dir", type=str, default=None,
                         help="If set, copy checkpoint.pt + losses.json here after EVERY epoch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    def log(*a, **kw):
        if accelerator.is_main_process:
            print(*a, **kw, flush=True)

    log("Using device:", device, "| num processes:", accelerator.num_processes)

    out_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    ckpt_path = out_dir / "checkpoint.pt"

    vae, normalizer, vae_ckpt = load_frozen_vae(args.vae_checkpoint, device)
    log(f"Loaded frozen VAE (latent_dim={vae.latent_dim}, fold={vae_ckpt['fold']})")

    genes = load_gene_panel(args.data_dir, CANCER_TYPE)
    assert genes == vae_ckpt["genes"], "gene panel mismatch between VAE checkpoint and current data-dir"

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    assert train_ids == vae_ckpt["train_sample_ids"], (
        "train split WSIs differ from the ones the VAE was fit on -- use the same --fold"
    )

    log("Building patch+expression index ...")
    index = build_patch_expr_index(args.data_dir, train_ids, genes, CANCER_TYPE)
    log(f"Total matched (image, expression) pairs: {len(index)}")
    if not index:
        raise RuntimeError(f"No patches found under {args.data_dir} -- check that the dataset was staged correctly.")

    dataset = HestPatchExprPairDataset(
        index, normalizer, image_size=args.image_size, low_res_size=args.low_res_size,
        cond_aug_max_std=args.cond_aug_max_std,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = RnaSRConditionedUnet(vae, image_size=args.image_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    loss_fn = nn.MSELoss()
    # Only the UNet has trainable params -- the VAE is frozen inside RnaSRConditionedUnet.
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    start_epoch = 0
    losses = []
    resume_from = args.resume or (ckpt_path if ckpt_path.exists() else None)
    if resume_from:
        log(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        accelerator.unwrap_model(model).model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt.get("losses", [])

    if start_epoch >= args.epochs:
        log(f"Already trained to epoch {start_epoch} >= target {args.epochs}. Nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        for step, (low_res, high_res, expr_z) in enumerate(loader):
            noise = torch.randn_like(high_res)
            timesteps = torch.randint(0, 1000, (high_res.shape[0],), device=device).long()
            noisy_high_res = noise_scheduler.add_noise(high_res, noise, timesteps)

            pred = model(noisy_high_res, low_res, timesteps, expr_z)
            loss = loss_fn(pred, noise)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            gathered_loss = accelerator.gather(loss.detach()).mean().item()
            epoch_losses.append(gathered_loss)
            if step % 50 == 0:
                log(f"  epoch {epoch} step {step}/{len(loader)} loss {gathered_loss:.5f}")

        losses.extend(epoch_losses)
        avg = sum(epoch_losses) / len(epoch_losses)
        dt = time.time() - t0
        log(f"Finished epoch {epoch}. Avg loss: {avg:.5f}. Time: {dt/60:.1f} min")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Only save the UNet (model.model) -- the VAE is a fixed,
            # separately versioned checkpoint (--vae-checkpoint).
            save_checkpoint(ckpt_path, accelerator.unwrap_model(model).model, optimizer, epoch, losses)
            losses_path = out_dir / "losses.json"
            with open(losses_path, "w") as f:
                json.dump(losses, f)

            if args.persist_dir:
                persist_dir = Path(args.persist_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(ckpt_path, persist_dir / "checkpoint.pt")
                shutil.copy(losses_path, persist_dir / "losses.json")
                log(f"  Synced checkpoint to {persist_dir} (epoch {epoch})")
        accelerator.wait_for_everyone()

    log("Training complete.")


if __name__ == "__main__":
    main()
