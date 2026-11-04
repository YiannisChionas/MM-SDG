# HEST Conditional Diffusion — HPC Experiment

Class-conditioned DDPM (cancer type: CCRCC, COAD, HCC, IDC, LUNG, LYMPH_IDC,
PAAD, PRAD, READ, SKCM — 10 classes), trained on pooled HEST-bench patches,
then evaluated via FID against a matched real-patch sample. Built to run on
the HUA Informatics HPC cluster per
[hpc-cluster-guide](https://gitlab.hua.gr/hpc/hpc-cluster-guide).

## Why this design

- **Lazy data loading** (`data_utils.HestPatchDataset`): the pooled dataset
  (~236K patches, ~35GB raw) is not preloaded into RAM like in the
  exploratory notebooks — it reads each `.h5` file on demand, since we can't
  assume a compute node has 35GB+ free.
- **Auto-resuming training/sampling**: the cluster enforces a **5h job time
  limit**. `train.py` resumes from `checkpoint.pt` automatically; `sample.py`
  skips classes/images already on disk. Both can simply be resubmitted after
  a timeout.
- **1000 synthetic + 1000 real images per class** (10,000 total each) —
  every cancer type here has at least ~3,000 real patches available, so this
  target doesn't require oversampling any class (unlike per-WSI conditioning,
  where some WSIs have only ~1,100 patches).

## 1. One-time setup on the cluster

```bash
ssh <username>@10.100.59.229

# Conda env (cluster runs CUDA 11.8 — note this differs from local dev, which used cu128)
conda create --name sdg python=3.11
conda activate sdg
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install diffusers h5py pillow numpy pytorch-fid huggingface_hub
```

### 1.1 Get the code (git clone)

Code lives in [MM-SDG](https://github.com/YiannisChionas/MM-SDG) — make sure
your local commits are pushed first (`git push` from this machine), then on
the cluster:

```bash
cd ~
git clone https://github.com/YiannisChionas/MM-SDG.git hest-cdm-repo
ln -s ~/hest-cdm-repo/hpc_experiment ~/hest-cdm
```

The `job_*.sh` scripts call `~/hest-cdm/train.py` etc., so the symlink keeps
them working regardless of where the actual repo checkout lives. To update
later: `cd ~/hest-cdm-repo && git pull`.

### 1.2 Get the dataset (download directly on the cluster)

Per the guide, dataset downloads should happen via an interactive session or
a non-GPU job — not on the login node directly, and not by uploading ~40GB
from your laptop. See section 2 below for the exact commands.

### 1.3 Pre-warm the FID Inception weights

Compute nodes may lack internet access, so fetch these once while you have
an interactive session open anyway:

```bash
srun --partition=yoda --nodes=1 --ntasks-per-node=1 --gpus-per-task=0 --time=00:15:00 --pty bash -i
conda activate sdg
python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3])"
exit
```

## 2. Download the dataset on the cluster

`sbatch -o/-e` below need the log directory to already exist:

```bash
mkdir -p ~/experiments/hest_cdm
```

The dataset (HEST-1k `hest-bench`, ~40GB) is downloaded **from inside the
cluster** straight to persistent storage — never uploaded from your laptop.

**One-time: authenticate with Hugging Face** (same account/token as local
dev; needed since `hest-bench` requires accepting its terms). This only
needs an interactive non-GPU session, and the resulting token is cached
under `~/.cache/huggingface/`, which is on your (shared) home directory —
so later batch jobs can reuse it without logging in again:

```bash
srun --partition=yoda --nodes=1 --ntasks-per-node=1 --gpus-per-task=0 --time=00:10:00 --pty bash -i
conda activate sdg
hf auth login
exit
```

**Submit the actual download as a (non-GPU) job**, so it survives
disconnects and doesn't run on the login node:

```bash
sbatch -J hest_download -o ~/experiments/hest_cdm/download.out -e ~/experiments/hest_cdm/download.err \
    slurm/job_download_data.sh
```

Check progress: `squeue -u $(whoami)` and `tail -f ~/experiments/hest_cdm/download.out`.
This lands at `/opt/storage/users/${USER}/hest-bench` — the `job_*.sh`
training/sampling/FID scripts stage it to fast local scratch
(`/var/local/storage/${USER}`) automatically per job, so you don't need to
move it again.

## 3. Train

Single job (10 epochs, ~3.2h estimated, fits under the 5h limit):

```bash
sbatch -J hest_cdm -o ~/experiments/hest_cdm/train.out -e ~/experiments/hest_cdm/train.err \
    slurm/job_train.sh --epochs 10
```

For more epochs than fit in one job, use the dependent multi-job submitter
instead (edit `N_STEPS`/`EPOCHS_PER_STEP` in the script first):

```bash
./slurm/submit_train.sh
```

Check progress: `squeue -u $(whoami)`, and tail the `.err`/`.out` log files
under `~/experiments/hest_cdm_train/logs/`.

## 4. Generate synthetic samples

Once `checkpoint.pt` exists under
`/opt/storage/users/${USER}/experiments/hest-cdm/`:

```bash
sbatch -J hest_cdm_sample -o ~/experiments/hest_cdm/sample.out -e ~/experiments/hest_cdm/sample.err \
    slurm/job_sample.sh
```

~2.1h estimated for 10,000 images at batch=128. If it times out, just
resubmit the same command — it resumes from where it left off.

## 5. Compute FID (real vs. synthetic)

```bash
sbatch -J hest_cdm_fid -o ~/experiments/hest_cdm/fid.out -e ~/experiments/hest_cdm/fid.err \
    slurm/job_fid.sh
```

Writes `fid_results.json` with a per-cancer-type FID plus a `POOLED`
(all-classes-together) FID to
`/opt/storage/users/${USER}/experiments/hest-cdm/fid_results.json`.

## Tuning knobs

| Flag | Default | Notes |
|---|---|---|
| `train.py --epochs` | 10 | cumulative target, not per-call |
| `train.py --batch-size` | 64 | matches local benchmark used for time estimates |
| `train.py --lr` | 1e-4 | **do not raise to 1e-3** — that caused the model to collapse to predicting zero noise (MSE loss stuck at ~1.0) in the local CCRCC/WSI-conditioned run |
| `sample.py --per-class` | 1000 | → 10,000 total; raise only if the FID result looks noisy and you have GPU-hours to spare |
| `compute_fid.py --per-class` | 1000 | should match `sample.py --per-class` |

## Storage cleanup

Per the cluster's storage policy, delete `/var/local/storage/${USER}/...`
scratch copies once a job finishes and results are confirmed copied back to
`/opt/storage/users/${USER}/experiments/hest-cdm/`. Files may be deleted by
administrators without warning, so keep only what's needed for the active
experiment.
