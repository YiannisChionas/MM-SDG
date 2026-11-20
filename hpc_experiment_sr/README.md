# HEST Cascaded Super-Resolution (Stage 2) — 224×224

Second stage of a cascaded pipeline on top of [hpc_experiment](../hpc_experiment)'s
64×64 class-conditioned model: conditions on **both** a low-res image AND
the cancer-type class label, to produce the native 224×224 HEST patch
resolution. Mirrors the paper's two-stage DDPM architecture (low-res →
super-resolution).

## Why this exists

Reviewing `hpc_experiment`'s FID (CCRCC=184.4, POOLED=137.3) surfaced a
structural issue: that experiment trains/generates at 64×64, and
`pytorch-fid` has to upsample 64→299 for InceptionV3 -- a **4.67x**
blow-up that loses/distorts information neither real nor synthetic images
actually have at that resolution. The paper works natively at 256×256
(only 1.17x to Inception's 299). Part of our FID gap (137-184 vs the
paper's 23.36) is likely this resolution handicap, not just model
size/training steps.

This experiment produces images at **224×224** (HEST's native patch size,
no resize needed for real patches) so FID evaluation only needs a
**1.33x** resize -- much closer to the paper's protocol, and a fairer test
of whether our generation quality (not just our evaluation pipeline) is
the bottleneck.

## Architecture

`SRConditionedUnet` (`data_utils_sr.py`): UNet2DModel conditioned on BOTH:
- a low-res image (3 channels, concatenated) -- during training, a real
  patch downsampled to 64×64 then upsampled back to 224×224 (standard
  cascaded-diffusion training practice: train stage 2 on artificially
  degraded real images, since stage 1's actual outputs are only needed at
  *sampling* time, not training time)
- the cancer-type class embedding (4 channels, broadcast) -- same
  mechanism as `hpc_experiment`'s `ClassConditionedUnet`

`in_channels = 3 (noisy high-res) + 3 (low-res cond) + 4 (class) = 10`.

**Deliberately smaller channel width (64 base) than hpc_experiment's 64×64
model (128 base)** despite the higher resolution -- mirrors the paper's own
strategy (their stage-2 UNet dim=128 vs stage-1's 256). Confirmed via local
benchmarking this is the only configuration that fits a realistic cluster
time budget at 224×224 (see below).

## Conditioning augmentation (added after the first FID result)

The first trained model's FID at 224×224 (282.86) was *worse* than at
64×64 (184.4) -- see this experiment's original result discussion.
A likely cause: training only ever shows stage 2 a **perfectly clean**
downsample→upsample degradation of a real image, but at sampling time
it's conditioned on stage 1's actual DDPM output, which carries its own
artifacts -- a train/inference distribution mismatch. `HestPatchPairDataset`
now adds random-strength Gaussian noise (std uniform in `[0, --cond-aug-max-std]`,
default 0.2) to the low-res conditioning during training, following
Ho et al. ("Cascaded Diffusion Models") and Imagen's noise conditioning
augmentation. Set `--cond-aug-max-std 0` to recover the old (unaugmented)
behaviour. Retraining with this change is still pending.

## Local benchmarks (before committing to this design)

Training (batch + backward), RTX 5070 Ti:

| Config | Params | Batch | ms/step | VRAM |
|---|---|---|---|---|
| 64ch base (chosen) | 28.5M | 8 | 177ms | 7.2GB |
| 64ch base | 28.5M | 16 | 415ms | 14.0GB |
| 96ch base | 64.0M | 8 | 340ms | 11.1GB |

Sampling (forward-only):

| Batch | ms/step | VRAM |
|---|---|---|
| 8 | 61.4ms | 1.4GB |
| 16 | 120.1ms | 2.6GB |
| 32 | 236.9ms | 5.1GB |

At the chosen config: **~87min/epoch** on the full pooled dataset (236,495
patches) for training; **~2.1h** for 1000 sampled images (100/class) at
batch=32.

## Time budget: why this needs 4 chained jobs

10-12 epochs at ~87min/epoch ≈ 14.5-17.4h, far over the 5h job limit.
`slurm/submit_train.sh` splits this into **4 sequential jobs of 3 epochs
each** (reaching 12 total), ~4.35h/job, fitting the 4h45m time limit with
margin. `train_sr.py` auto-resumes from `checkpoint.pt`, same pattern as
every other experiment here.

## Cascaded sampling

`sample_sr.py` loads **two** checkpoints:
1. `hpc_experiment`'s already-trained `checkpoint.pt` (`Stage1ClassConditionedUnet`,
   duplicated in `data_utils_sr.py` so this folder stays self-contained) —
   generates the 64×64 low-res image
2. This experiment's `checkpoint.pt` (`SRConditionedUnet`) — refines the
   upsampled low-res image + class label into the final 224×224 output

Both reverse-diffusion passes (1000 steps each) run per batch, so this is
slower per-image than a single-stage model -- budgeted at 100/class
(1000 total) rather than `hpc_experiment`'s 1000/class, same reduced-scale
caveat as `hpc_experiment_large` (FID comparison across experiments with
different sample sizes isn't apples-to-apples; see that experiment's
README for the full explanation).

## Usage

```bash
ln -s ~/git/MM-SDG/hpc_experiment_sr ~/hest-sr
mkdir -p ~/experiments/hest_sr

# Training (multi-job -- required, won't fit in one 5h job)
./slurm/submit_train.sh

# Sampling (once this experiment's checkpoint.pt exists AND
# hpc_experiment's checkpoint.pt already exists from that earlier run)
sbatch -J hest_sr_sample -o ~/experiments/hest_sr/sample.out -e ~/experiments/hest_sr/sample.err \
    slurm/job_sample.sh

# FID
sbatch -J hest_sr_fid -o ~/experiments/hest_sr/fid.out -e ~/experiments/hest_sr/fid.err \
    slurm/job_fid.sh
```

Results land in `/opt/storage/users/${USER}/experiments/hest-sr/`
(`checkpoint.pt`, `losses.json`, `synthetic/`, `fid_results.json`).
