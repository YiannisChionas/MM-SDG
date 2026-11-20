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
| **This experiment** | **configurable, default 17,655** | **configurable, default 6,000 → 4,000** | **configurable, default 200** | **~282M at defaults** |

## Important difference from the paper: gene panel identity

The paper trains on **bulk RNA-seq** across ~17,655 genes from TCGA-style
patient-level sequencing. We only have HEST-bench's **spatial**
transcriptomics (per-spot, not per-patient), and there is no published
list of exactly which 17,655 genes the paper used. So this experiment
selects its **own** 17,655-gene panel directly from our data
(`select_genes.py`): highly-variable-gene selection (by variance of
library-size-normalized + log1p expression) over all ~36,601 genes
available in the CCRCC train-split WSIs' full transcriptome, after
dropping genes detected in under 1% of spots. This matches the paper's
panel **size**, not its exact identity — an explicit, documented
divergence, not an oversight.

## Architecture

`BetaVAEPaper` (`data_utils_vae_paper.py`): same MLP beta-VAE shape as
`hpc_experiment_rna`'s `BetaVAE`, but with independently configurable
`hidden1`/`hidden2`/`latent_dim` (defaults 6000/4000/200, vs. the single
`hidden_dim` the small VAE halves internally). `vae_loss` (recon MSE + β·KL)
is unchanged.

## Pipeline

```bash
ln -s ~/git/MM-SDG/hpc_experiment_vae_paper ~/hest-vae-paper
mkdir -p ~/experiments/hest_vae_paper

# Step 1: select the 17,655-gene panel (CPU-only, run once)
sbatch -J hest_vae_paper_genes -o ~/experiments/hest_vae_paper/genes.out -e ~/experiments/hest_vae_paper/genes.err \
    slurm/job_select_genes.sh

# Step 2: train the VAE (NOT submitted yet -- see "Settings to decide
# together" below first)
sbatch -J hest_vae_paper_train -o ~/experiments/hest_vae_paper/train.out -e ~/experiments/hest_vae_paper/train.err \
    slurm/job_train_vae_paper.sh --epochs 50 --batch-size 256 --beta 0.01 --lr 1e-4
```

Results land in `/opt/storage/users/${USER}/experiments/hest-vae-paper/`.

## Settings to decide together before running at scale

This is a genuinely different regime from every other experiment in this
repo so far -- a ~282M-param model trained on only ~56K training spots
(our full CCRCC train-split size), vs. the paper's presumably much larger
patient cohort pooled across twelve cancer types. Things worth deciding
before submitting `job_train_vae_paper.sh` for real:

1. **Overfitting risk**: the first encoder layer alone is
   `17655 × 6000 ≈ 106M` weights, trained on ~56K examples. May need
   stronger regularization (weight decay, dropout) or a smaller
   hidden/latent size than the paper's, even though the point was to
   match paper scale.
2. **β (posterior collapse)**: `hpc_experiment_rna`'s VAE needed β=0.01
   (not the paper's reported β≈0.005) to avoid posterior collapse at a
   much smaller scale -- unclear whether that transfers to 200 latent
   dims / this much larger capacity. Worth monitoring KL alongside recon
   loss from epoch 0, same diagnostic as before.
3. **Batch size / GPU**: `--batch-size 256` is just inherited from the
   small VAE's default and hasn't been benchmarked at this param count.
   Depending on measured VRAM, this may need yoda's 40GB A100 at minimum,
   or the anakin/luke access requested separately (see repo root).
4. **Epochs**: 50 is the small VAE's default, not benchmarked here either
   -- convergence behavior at this scale is unknown until we run it.
5. **Gene panel size**: literally matching 17,655 may not be the right
   choice given our much smaller spot count -- worth considering a
   smaller panel (e.g. top 2,000-5,000 HVGs) as a more realistic middle
   ground between the existing 50-gene panel and the paper's scale.

None of these are blocking issues with the code -- `select_genes.py` and
`train_vae_paper.py` both work end-to-end (see smoke test) at any
hidden/latent/top-k size via CLI args. They're open modeling decisions to
make together before committing cluster time to a real run.
