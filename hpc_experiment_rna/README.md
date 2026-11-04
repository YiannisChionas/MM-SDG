# RNA-Conditioned Diffusion — HPC Experiment (CCRCC)

Two-stage pipeline, matching the paper's approach: (1) train a beta-VAE on
real gene expression to learn a compact latent embedding, (2) condition the
diffusion UNet on that embedding instead of a discrete class label. Scoped
to **CCRCC only** for now (consistent 50-gene panel, single cancer type) —
see [hpc_experiment/](../hpc_experiment) for the pooled 10-cancer-type,
class-conditioned version this builds on.

Uses the **official HEST-bench WSI-level train/test split** (fold 0: 20
train WSIs / 4 held-out test WSIs) so evaluation happens on gene-expression
profiles the model never saw during training — not just unseen patches from
WSIs it already trained on.

## Why this design

- **Conditioning signal**: each patch's own real gene expression (50-gene
  `var_50genes` panel, library-size-normalized + log1p), encoded through a
  frozen beta-VAE's posterior mean into a 16-dim latent, broadcast into the
  UNet as extra input channels — same broadcast-and-concat mechanism as the
  class-conditioned version, but continuous and per-patch instead of
  discrete and per-WSI/per-cancer-type.
- **Why a beta-VAE and not raw expression directly**: compresses 50
  (correlated, noisy) genes into a smoother, lower-dimensional latent space
  before using it as a conditioning signal — per the paper, this produced
  better results than conditioning on raw expression.
- **Paired evaluation**: `sample_rna.py` generates one synthetic image per
  held-out real spot, conditioned on that spot's *actual* expression, and
  saves both into same-named files under `real/` and `synthetic/`. FID is
  computed between the two folders.
- **Auto-resuming** at every stage (VAE training is fast enough not to need
  it, but diffusion training and sampling do), same pattern as
  `hpc_experiment/`.

## 1. Setup

Same conda env as `hpc_experiment/` (add `anndata` and `pandas`):

```bash
conda activate sdg
pip install anndata pandas
```

Code: same repo, just a different subfolder. Symlink it separately so the
`~/hest-rna/...` paths in the slurm scripts resolve:

```bash
ln -s ~/hest-cdm-repo/hpc_experiment_rna ~/hest-rna
```

Needs the same `hest-bench` dataset already downloaded per
`hpc_experiment/README.md` §2 (`/opt/storage/users/${USER}/hest-bench`) —
no separate download step.

## 2. Train the beta-VAE (stage 1)

```bash
mkdir -p ~/experiments/hest_rna
sbatch -J hest_rna_vae -o ~/experiments/hest_rna/vae.out -e ~/experiments/hest_rna/vae.err \
    slurm/job_train_vae.sh
```

Fast (~minutes, not hours) — 50 epochs over a ~50-dim MLP on ~60K spots.
Produces `vae_checkpoint.pt` under
`/opt/storage/users/${USER}/experiments/hest-rna/`, which bakes in the gene
list, the beta-VAE weights, and the train-split normalization stats (mean/std)
so stage 2 and sampling reuse exactly the same preprocessing.

## 3. Train the diffusion model (stage 2)

Requires `vae_checkpoint.pt` from step 2.

```bash
sbatch -J hest_rna_train -o ~/experiments/hest_rna/train.out -e ~/experiments/hest_rna/train.err \
    slurm/job_train_rna.sh --epochs 10
```

For more epochs than fit in one 5h job: `./slurm/submit_train_rna.sh`
(edit `N_STEPS`/`EPOCHS_PER_STEP` first).

## 4. Generate paired samples on held-out WSIs

```bash
sbatch -J hest_rna_sample -o ~/experiments/hest_rna/sample.out -e ~/experiments/hest_rna/sample.err \
    slurm/job_sample_rna.sh
```

Default cap: 2000 held-out spots (fold-0 test WSIs have more than that
combined — raise `--max-samples` if you have GPU-hours to spare). Resumable.

## 5. Compute FID

```bash
sbatch -J hest_rna_fid -o ~/experiments/hest_rna/fid.out -e ~/experiments/hest_rna/fid.err \
    slurm/job_fid_rna.sh
```

Writes `fid_rna_results.json` to
`/opt/storage/users/${USER}/experiments/hest-rna/`.

## Comparing against the class-conditioned experiment

Both pipelines write their results as flat JSON, so once you have:

- `hpc_experiment`'s `fid_results.json` → `POOLED` key (all-cancer-type,
  class-conditioned)
- this experiment's `fid_rna_results.json` → `CCRCC_rna_conditioned`

...they're directly comparable **only informally**, since the class-
conditioned FID pools 10 cancer types while this one is CCRCC-only. For a
fair head-to-head, also run `hpc_experiment`'s pipeline restricted to CCRCC
alone (or extend this experiment to pooled cancer types later — see note in
`hpc_experiment`'s README about per-cancer-type gene panel mismatches if you
do).

## Tuning knobs

| Flag | Default | Notes |
|---|---|---|
| `train_vae.py --latent-dim` | 16 | conditioning vector size fed into the UNet (replaces `class_emb_size=4` from the discrete version) |
| `train_vae.py --beta` | 4.0 | higher = more disentangled/compressed latent, at some reconstruction cost |
| `train_rna.py --lr` | 1e-4 | same caution as `hpc_experiment`: raising this collapsed training there |
| `sample_rna.py --max-samples` | 2000 | held-out spots to generate; raise if you want a tighter FID estimate |

## Storage cleanup

Same policy as `hpc_experiment/README.md` — clean up
`/var/local/storage/${USER}/...` scratch copies once results are confirmed
copied back to `/opt/storage/users/${USER}/experiments/hest-rna/`.
