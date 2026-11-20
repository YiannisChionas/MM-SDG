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

## Architecture

`BetaVAEPaper` (`data_utils_vae_paper.py`): same MLP beta-VAE shape as
`hpc_experiment_rna`'s `BetaVAE`, but with independently configurable
`hidden1`/`hidden2`/`latent_dim` (defaults 6000/4000/200, vs. the single
`hidden_dim` the small VAE halves internally). `vae_loss` (recon MSE + β·KL)
is unchanged. At 14,166 input genes the model is **~220.4M params**
(measured directly, not estimated) -- much closer to the paper's ~282M
than the earlier (abandoned) HVG-filtered estimate of ~56M, and the
overfitting-risk discussion below should be read with that in mind.

## Pipeline

```bash
ln -s ~/git/MM-SDG/hpc_experiment_vae_paper ~/hest-vae-paper
mkdir -p ~/experiments/hest_vae_paper

# Step 1: select the gene panel (CPU-only, run once; installs scanpy
# into the sdg env on first run since it isn't there yet)
sbatch -J hest_vae_paper_genes -o ~/experiments/hest_vae_paper/genes.out -e ~/experiments/hest_vae_paper/genes.err \
    slurm/job_select_genes.sh

# Step 2: train the VAE (NOT submitted yet -- see "Settings to decide
# together" below first)
sbatch -J hest_vae_paper_train -o ~/experiments/hest_vae_paper/train.out -e ~/experiments/hest_vae_paper/train.err \
    slurm/job_train_vae_paper.sh --epochs 50 --batch-size 256 --beta 0.01 --lr 1e-4
```

`job_train_vae_paper.sh` picks up whichever `gene_panel_*.json` exists in
the experiment dir, so it works regardless of the exact gene count.
Results land in `/opt/storage/users/${USER}/experiments/hest-vae-paper/`.

## Settings to decide together before running at scale

At ~220.4M params trained on ~56K training spots, this is a genuinely
different regime from every other experiment in this repo so far --
nearly a 4:1 param-to-example ratio. Things worth deciding before
submitting `job_train_vae_paper.sh` for real:

1. **Overfitting risk**: real and significant at this ratio. Likely
   needs weight decay and/or dropout; may still be worth considering a
   smaller `hidden1`/`hidden2` (not the gene panel) if early runs show
   clear overfitting (train loss falling, held-out reconstruction not).
2. **β (posterior collapse)**: `hpc_experiment_rna`'s VAE needed β=0.01
   (not the paper's reported β≈0.005) to avoid posterior collapse at a
   much smaller scale -- unclear whether that transfers to 200 latent
   dims / this capacity. Worth monitoring KL alongside recon loss from
   epoch 0, same diagnostic as before.
3. **Batch size / GPU**: `--batch-size 256` is just inherited from the
   small VAE's default and hasn't been benchmarked at this param count.
   At ~220M params this is back in anakin/luke territory (see repo root
   README for that access request) rather than comfortably fitting yoda
   -- needs an actual benchmark run to confirm either way.
4. **Epochs**: 50 is the small VAE's default, not benchmarked here either
   -- convergence behavior at this scale is unknown until we run it.

None of these are blocking issues with the code -- `select_genes.py` and
`train_vae_paper.py` both work end-to-end (smoke-tested) via CLI args.
They're open modeling decisions to make together before committing
cluster time to a real run.
