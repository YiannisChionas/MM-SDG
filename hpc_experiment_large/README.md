# HEST Class-Conditioned Diffusion — LARGE Model Capacity Ablation

Model-capacity ablation on top of [hpc_experiment](../hpc_experiment): same
class-conditioned (cancer type, pooled 10 types) pipeline, same data, same
hyperparameters -- only the UNet is ~4x bigger. Motivated by the gap between
our baseline FID (pooled=137.3, CCRCC=184.4) and the RNA-CDM paper's
FID50k=23.36: the paper uses a 1.8B-param cascaded model vs our 114M-param
single-stage one, and we hadn't actually tested whether capacity explains
any of that gap.

## What changed vs hpc_experiment

Only `data_utils.py`'s `ClassConditionedUnet.block_out_channels`:
`(128,128,256,256,512,512)` → `(256,256,512,512,1024,1024)`.

| | Small (hpc_experiment) | Large (this folder) |
|---|---|---|
| Params | 113.7M | **454.5M** (~4x) |
| Everything else | identical | identical |

`train.py`, `sample.py`, `compute_fid.py` are unmodified copies (they're
architecture-agnostic -- all they do is import whatever `ClassConditionedUnet`
is defined in the local `data_utils.py`). Kept as full copies rather than
importing from `hpc_experiment/` so each experiment folder stays
self-contained and can't silently drift if one changes later.

## A local benchmark surprise worth knowing about

First attempt to estimate the large model's speed locally (RTX 5070 Ti,
16GB) at batch=64 measured **18.6s/step -- a 62.7x slowdown** vs the small
model's 297ms/step. That number is wrong in a misleading way: it's a
**VRAM-exhaustion artifact**, not real compute cost. A batch-size sweep
confirmed it:

| Batch size | Step time | Peak VRAM |
|---|---|---|
| 8 | 167ms | 9.14GB |
| 16 | 262ms | 10.13GB |
| 32 | 535ms | 14.75GB (16GB card, nearly maxed) |
| 64 | **18,652ms** | (OOM-adjacent -- catastrophic paging) |

Extrapolating the clean batch=8→32 trend to batch=64 linearly gives
**~1070ms/step**, i.e. **~3.6x slower** than the small model at the same
batch size -- in line with what 4x the parameters should actually cost.
The cluster's A100 40GB has more than double the VRAM headroom of the local
card, so batch=64 should not hit this wall there -- but this is
**unverified at cluster scale**. Watch the first epoch's timing in the
training job's log; if it's wildly slower than ~40min/epoch, the model may
be running into memory pressure there too and batch size should be lowered.

## Time budget consequences

- **Training**: ~40min/epoch estimated (11min × 3.6) → 10 epochs ≈ 6.7h,
  over the 5h job limit. Use `submit_train.sh` (5 epochs/job × 2 jobs)
  instead of a single `job_train.sh` call.
- **Sampling**: generating the full 10,000 images (1000/class, matching the
  small model) was estimated at >14h -- not feasible. Reduced to
  **200/class (2000 total)**, batch=64, estimated ~2.5h.
- **FID**: `--per-class 200` to match the reduced sample count.

## The resulting fairness caveat

Comparing this experiment's FID against `hpc_experiment`'s **is not
apples-to-apples on sample size**: small model used 1000/class (10,000
total), this one uses 200/class (2,000 total). FID is a known-biased
estimator at small N -- smaller samples tend to score *worse* (higher/more
biased FID) than larger ones, even for an identically-good generator. So if
the large model's FID comes out lower despite the smaller, more-biased
sample, that's a *conservative* result (the true gap is probably at least
that large). If it comes out higher, the sample-size confound makes it
ambiguous whether that's really worse generation or just FID bias --
wouldn't be safe to conclude "bigger model is worse" from that without
re-running at matched sample size.

## Usage

Same flow as `hpc_experiment/README.md` sections 3-5, pointed at this
folder's scripts:

```bash
ln -s ~/git/MM-SDG/hpc_experiment_large ~/hest-large
mkdir -p ~/experiments/hest_large

# Training (multi-job -- single job_train.sh call won't fit in 5h)
./slurm/submit_train.sh

# Sampling (once checkpoint.pt exists)
sbatch -J hest_large_sample -o ~/experiments/hest_large/sample.out -e ~/experiments/hest_large/sample.err \
    slurm/job_sample.sh

# FID
sbatch -J hest_large_fid -o ~/experiments/hest_large/fid.out -e ~/experiments/hest_large/fid.err \
    slurm/job_fid.sh
```

Results land in `/opt/storage/users/${USER}/experiments/hest-large/`
(`checkpoint.pt`, `losses.json`, `synthetic/`, `fid_results.json`), same
layout as `hpc_experiment`.
