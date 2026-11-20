# Paper-Scale β-VAE (CCRCC)

A beta-VAE matching the RNA-CDM paper's reported architecture scale, to
replace [hpc_experiment_rna](../hpc_experiment_rna)'s much smaller VAE
(50 genes / 128 hidden / 16 latent, ~19.4K params) as the gene-expression
embedding for a paper-scale diffusion cascade. This is step 1 of that
larger effort (the diffusion stages themselves are a separate, later
experiment once this VAE is validated).

| | Input genes | Encoder/decoder hidden | Latent dim | Params |
|---|---|---|---|---|
| `hpc_experiment_rna`'s VAE | 50 | 128 | 16 | ~19.4K |
| Paper (RNA-CDM) | 17,655 | 6,000 → 4,000 | 200 | ~282M |
| **This experiment** | **14,166** (see below) | **configurable, default 6,000 → 4,000** | **configurable, default 200** | **~220.4M at this gene count** |

## Gene panel: QC only, no HVG cutoff -- and why

The paper's 17,655 genes were themselves just whatever **their** own
preprocessing happened to leave them with ("these pre-processing steps
left us with a total of 17,655 genes") -- not a number they deliberately
targeted. Since there's no published list of exactly which genes those
are, and we only have **spatial** (per-spot) transcriptomics rather than
their **bulk** (per-patient) data, forcing our own panel to that exact
size would just be cosmetic. `select_genes.py` instead applies scanpy's
standard QC filter (`sc.pp.filter_genes`, min_cells as 1% of spots) and
nothing else.

**We tried an HVG dispersion cutoff first** (`flavor='seurat'`, scanpy's
common default thresholds) and measured how much it throws away: from
17,943 genes common to all 20 train WSIs (not the full ~36,601 -- several
WSIs are each missing some genes), `filter_genes` alone leaves 14,166
while keeping **99.9% of total raw UMI counts** (i.e. near-lossless --
the dropped genes are essentially undetected noise). The HVG cutoff on
top of that keeps only 2,913/14,166 genes, but those genes capture just
**22% of total expression variance** -- throwing away ~78% of the signal,
too lossy for a generative model's conditioning input (HVG selection is
standard for clustering/visualization, where you *want* to discard
redundant/low-signal genes, but that's not our use case here). So we
kept all 14,166 QC-passed genes and let the VAE's own bottleneck
(`latent_dim`) do the real compression -- the same philosophy as the
paper feeding its VAE a large, only lightly filtered gene set.
`select_qc_passed_genes` in `data_utils_vae_paper.py` still has the
numbers from both approaches in its docstring.

## Architecture and training recipe -- matched to the paper's Methods

`BetaVAEPaper` (`data_utils_vae_paper.py`): same MLP beta-VAE shape as
`hpc_experiment_rna`'s `BetaVAE`, but with independently configurable
`hidden1`/`hidden2`/`latent_dim` (defaults 6000/4000/200). `vae_loss`
(recon MSE + β·KL) is unchanged. At 14,166 input genes the model is
**~220.4M params** (measured directly, not estimated) -- much closer to
the paper's ~282M than the earlier (abandoned) HVG-filtered estimate of
~56M.

We confirmed the following directly from the paper's Methods
("βVAE for multi-cancer latent embedding generation") and matched them
as `train_vae_paper.py` defaults:
- **BatchNorm1d + LeakyReLU** between layers (was SiLU, no BatchNorm).
- **β = 0.005** (was 0.01, `hpc_experiment_rna`'s own empirically-tuned
  value at a much smaller scale -- untested whether 0.005 also avoids
  posterior collapse at this scale; monitor KL from epoch 0).
- **lr = 3e-3**, Adam (was 1e-4), with **linear warmup (`--warmup-epochs`,
  default 10) + cosine decay to 0** over the remaining epochs
  (`warmup_cosine_lambda` in `train_vae_paper.py`) -- matches the paper's
  "a warm-up and a cosine learning rate scheduler" (exact warmup length
  isn't stated in the paper, so 10 is our own choice). Scheduler state is
  checkpointed and resumes correctly (smoke-tested: LR rises through
  warmup, decays smoothly after, and continues the same curve across a
  resume rather than resetting).
- **250 max epochs with early stopping** (`--patience 20`, monitored on
  a validation split carved out of the train WSIs via `split_train_val_ids`
  -- separate from, and never touching, the official held-out test split).
  Best-val checkpoint saved separately as `vae_paper_best.pt`.
- **Batch size kept at our own default (256), not the paper's 128** --
  deliberate: smaller batches would underutilize whatever GPU we end up
  training on, with no corresponding benefit for us.

## Pipeline

```bash
ln -s ~/git/MM-SDG/hpc_experiment_vae_paper ~/hest-vae-paper
mkdir -p ~/experiments/hest_vae_paper

# Step 1: select the gene panel (CPU-only, run once; installs scanpy
# into the sdg env on first run since it isn't there yet)
sbatch -J hest_vae_paper_genes -o ~/experiments/hest_vae_paper/genes.out -e ~/experiments/hest_vae_paper/genes.err \
    slurm/job_select_genes.sh

# Step 2: train the VAE (defaults now match the paper's beta/epochs/
# architecture -- see above; NOT submitted yet, still waiting on an
# actual batch-size/partition benchmark, see below)
sbatch -J hest_vae_paper_train -o ~/experiments/hest_vae_paper/train.out -e ~/experiments/hest_vae_paper/train.err \
    slurm/job_train_vae_paper.sh
```

`job_train_vae_paper.sh` picks up whichever `gene_panel_*.json` exists in
the experiment dir, so it works regardless of the exact gene count.
Results land in `/opt/storage/users/${USER}/experiments/hest-vae-paper/`.

## Still open before running at scale

At ~220.4M params trained on ~56K training spots (minus the validation
split, so really ~42-45K), this is a genuinely different regime from
every other experiment in this repo so far:

1. **Overfitting risk**: real and significant at this ratio. The paper's
   only stated defenses are BatchNorm + early stopping (no dropout/weight
   decay mentioned) -- both now in place here too, but whether they're
   *enough* at our much smaller example count (likely fewer patients in
   their bulk cohort than our spot count, but spot-level data is noisier
   per-example) is untested until we actually run it and watch val loss.
2. **Batch size / GPU**: still not benchmarked at ~220M params. This is
   back in anakin/luke territory (see repo root README for that access
   request) rather than comfortably fitting yoda -- needs an actual
   benchmark run to confirm either way.

None of these are blocking issues with the code -- `select_genes.py` and
`train_vae_paper.py` both work end-to-end (smoke-tested, including the
early-stopping/resume logic) via CLI args. They're open questions that
need an actual cluster run to answer, not further guessing.
