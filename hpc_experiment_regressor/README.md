# Image → Gene-Expression Regressor (RNA-conditioning fidelity check)

An independent evaluation metric alongside FID: a small CNN trained to
predict a patch's gene expression vector (var_50genes panel) **from the
image alone**, on REAL (image, expression) pairs only. It is then applied
to RNA-conditioned synthetic images to check whether the conditioning
signal actually materializes visually, not just whether the image looks
generically realistic.

## Why this exists

FID measures overall realism, but says nothing about whether a specific
RNA-conditioned synthetic image actually encodes the expression vector it
was conditioned on. Two images can have identical FID contribution while
one is "generic realistic tissue" and the other faithfully reflects its
condition. This regressor is the **inverse** of the generative direction:

- Diffusion model: expression → image
- Regressor: image → expression

Trained only on real pairs, then run on synthetic images that were
generated conditioned on a known real `expr_target` (same paired protocol
as [hpc_experiment_rna](../hpc_experiment_rna)'s `sample_rna.py` and
[hpc_experiment_rna_sr](../hpc_experiment_rna_sr)'s `sample_rna_sr.py`).
Comparing the regressor's prediction on the synthetic image against
`expr_target` measures conditioning fidelity, independent of FID.

## Architecture

`ImageToExpressionRegressor` (`data_utils_regressor.py`): 4 conv blocks
(32→64→128→256 channels, stride 2 each) + adaptive avg pool + 2-layer MLP
head → 50-dim output (z-scored expression). Works at any input resolution,
so the same class trains both regressors below.

**Two separate checkpoints, matched to the resolution of what they'll
judge** (same resolution-matching discipline as the FID evaluations
elsewhere in this repo):
- `regressor_64.pt`: trained on patches resized to 64×64 — judges
  `hpc_experiment_rna`'s synthetic output.
- `regressor_224.pt`: trained on **native** 224×224 patches — judges
  `hpc_experiment_rna_sr`'s cascaded synthetic output.

Both trained on CCRCC train-split WSIs only (official HEST-bench fold 0),
same barcode-matched (image, expression) pairs as `hpc_experiment_rna`.

## Usage

```bash
ln -s ~/git/MM-SDG/hpc_experiment_regressor ~/hest-regressor
mkdir -p ~/experiments/hest_regressor

# Train both resolution-matched regressors
sbatch -J hest_regressor_train64 -o ~/experiments/hest_regressor/train64.out -e ~/experiments/hest_regressor/train64.err \
    slurm/job_train_regressor.sh --image-size 64 --epochs 15
sbatch -J hest_regressor_train224 -o ~/experiments/hest_regressor/train224.out -e ~/experiments/hest_regressor/train224.err \
    slurm/job_train_regressor.sh --image-size 224 --epochs 15

# Evaluate against hpc_experiment_rna's paired samples (64x64)
sbatch -J hest_regressor_eval64 -o ~/experiments/hest_regressor/eval64.out -e ~/experiments/hest_regressor/eval64.err \
    slurm/job_eval_regressor.sh \
    --regressor-checkpoint /opt/storage/users/$USER/experiments/hest-regressor/regressor_64.pt \
    --samples-dir /opt/storage/users/$USER/experiments/hest-rna/samples \
    --output-json /opt/storage/users/$USER/experiments/hest-regressor/eval_rna_64.json \
    --max-samples 2000

# Evaluate against hpc_experiment_rna_sr's paired cascaded samples (224x224)
sbatch -J hest_regressor_eval224 -o ~/experiments/hest_regressor/eval224.out -e ~/experiments/hest_regressor/eval224.err \
    slurm/job_eval_regressor.sh \
    --regressor-checkpoint /opt/storage/users/$USER/experiments/hest-regressor/regressor_224.pt \
    --samples-dir /opt/storage/users/$USER/experiments/hest-rna-sr/samples \
    --output-json /opt/storage/users/$USER/experiments/hest-regressor/eval_rna_sr_224.json \
    --max-samples 2000
```

`--max-samples` must match the `--max-samples` used when `sample_rna.py` /
`sample_rna_sr.py` generated `--samples-dir`, since `eval_regressor.py`
recomputes their exact deterministic (fold, seed) index to recover the
ground-truth expression vector behind each filename — it doesn't read the
expression vectors from disk (the sampling scripts only save images).

## What `eval_regressor.py` reports

- `real_baseline_mae`: regressor error on a fresh sample of real held-out
  patches — the upper bound on how well expression can be predicted from
  an image at all, independent of generation.
- `real_pairs_mae`: regressor error on the exact `real/` copies saved
  alongside the synthetic images — a sanity check, should be close to
  `real_baseline_mae` (confirms the eval pipeline itself, independent of
  any generative model).
- `synthetic_mae`: regressor error on the `synthetic/` images vs the same
  target expression they were conditioned on — **the actual fidelity
  number**. All MAE values are in z-scored units (comparable across the
  two resolutions).

Results land in `/opt/storage/users/${USER}/experiments/hest-regressor/`.

## Result

*(to be filled in once both regressors are trained and evaluated)*

| | real_baseline_mae | real_pairs_mae | synthetic_mae |
|---|---|---|---|
| `hpc_experiment_rna` (64×64) | | | |
| `hpc_experiment_rna_sr` (224×224 cascade) | | | |

If `synthetic_mae` stays close to `real_baseline_mae`/`real_pairs_mae`,
the RNA-conditioning signal is faithfully materialized in the generated
image, not just "generically realistic." A `synthetic_mae` well above the
real baseline would mean the diffusion model produces plausible-looking
tissue largely independent of the specific expression vector it was given
— i.e. good FID without real conditioning fidelity.
