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
| **This experiment** | **2,913** (see below) | **configurable, default 6,000 → 4,000** | **configurable, default 200** | **~56M at this gene count** |

## Gene panel: standard method, not a forced target

The paper's 17,655 genes were themselves just whatever **their** own
preprocessing happened to leave them with ("these pre-processing steps
left us with a total of 17,655 genes") -- not a number they deliberately
targeted. Since there's no published list of exactly which genes those
are, and we only have **spatial** (per-spot) transcriptomics rather than
their **bulk** (per-patient) data, forcing our own panel to that exact
size would just be cosmetic. Instead, `select_genes.py` applies scanpy's
**standard** highly-variable-gene procedure (`flavor='seurat'`, the
field's common default: `sc.pp.filter_genes` + `sc.pp.highly_variable_genes`
with default `min_mean=0.0125, max_mean=3, min_disp=0.5`) to the pooled
CCRCC train-split spots, and reports whatever gene count results.

**Result on our data**: starting from 17,943 genes common to all 20
train WSIs (not the full ~36,601 -- several WSIs are each missing some
genes), `filter_genes(min_cells=1%)` leaves 14,166, and the standard HVG
threshold selects **2,913** -- noticeably *smaller* than the paper's
17,655, not larger. This isn't a choice; it's what the standard
procedure yields on our noisier, sparser spot-level data (few UMIs/spot,
lots of zeros) versus the paper's deep, pooled bulk sequencing. A fully
honest, non-cherry-picked number.

## Architecture

`BetaVAEPaper` (`data_utils_vae_paper.py`): same MLP beta-VAE shape as
`hpc_experiment_rna`'s `BetaVAE`, but with independently configurable
`hidden1`/`hidden2`/`latent_dim` (defaults 6000/4000/200, vs. the single
`hidden_dim` the small VAE halves internally). `vae_loss` (recon MSE + β·KL)
is unchanged. At 2,913 input genes (vs. the paper's 17,655), the encoder's
first layer is `2913 × 6000 ≈ 17.5M` weights instead of `17655 × 6000 ≈ 106M`
-- total model ~56M params, not ~282M -- which meaningfully changes the
overfitting-risk discussion below.

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

With the gene count now at 2,913 (not 17,655), the overfitting-risk
picture is less extreme than originally worried, but there are still
open decisions before submitting `job_train_vae_paper.sh` for real:

1. **Overfitting risk**: ~56M params trained on ~56K spots is still a
   roughly 1:1 param-to-example ratio -- may still want weight decay or a
   smaller hidden/latent size, though far less dire than at the original
   17,655-gene/282M-param estimate.
2. **β (posterior collapse)**: `hpc_experiment_rna`'s VAE needed β=0.01
   (not the paper's reported β≈0.005) to avoid posterior collapse at a
   much smaller scale -- unclear whether that transfers to 200 latent
   dims / this capacity. Worth monitoring KL alongside recon loss from
   epoch 0, same diagnostic as before.
3. **Batch size / GPU**: `--batch-size 256` is just inherited from the
   small VAE's default and hasn't been benchmarked at this param count.
   At ~56M params this is far less likely to need anakin/luke than the
   original 282M estimate -- yoda is probably sufficient, to be confirmed
   by an actual benchmark run.
4. **Epochs**: 50 is the small VAE's default, not benchmarked here either
   -- convergence behavior at this scale is unknown until we run it.

None of these are blocking issues with the code -- `select_genes.py` and
`train_vae_paper.py` both work end-to-end (smoke-tested) via CLI args.
They're open modeling decisions to make together before committing
cluster time to a real run.
