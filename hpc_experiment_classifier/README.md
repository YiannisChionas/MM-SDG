# Image → Cancer-Type Classifier (class-conditioning fidelity check)

The discrete-label analogue of
[hpc_experiment_regressor](../hpc_experiment_regressor)'s
image→expression regressor: an independent CNN trained to predict cancer
type **from the image alone**, on real patches only, then applied to
class-conditioned synthetic images to check whether the conditioning
signal (the cancer-type label) actually materializes visually — the
"Regressor Alignment" evaluation metric, in its classification form.

## Why this exists

Same motivation as `hpc_experiment_regressor`: FID measures overall
realism but says nothing about whether a specific synthetic image
actually depicts the cancer type it was conditioned on. This is the
missing symmetric check for the class-conditioned side of the
RNA-vs-class-conditioning comparison — without it, only the RNA side had
a conditioning-fidelity metric alongside FID.

- Diffusion model: class label → image
- Classifier: image → class label (the inverse direction)

## Architecture

`ImageToClassClassifier` (`data_utils_classifier.py`): same 4-conv-block
backbone as the regressor's `ImageToExpressionRegressor`, softmax head
over the 10 cancer types instead of a 50-dim regression head. Works at
any input resolution.

**Two resolution-matched checkpoints**:
- `classifier_64.pt`: trained on patches resized to 64×64 — judges
  `hpc_experiment`'s synthetic output.
- `classifier_224.pt`: trained on **native** 224×224 patches — judges
  `hpc_experiment_sr`'s cascaded synthetic output.

## Train/test split caveat

`hpc_experiment` / `hpc_experiment_sr` train their diffusion models on
**all** WSIs with no held-out split (unlike the RNA experiments, which use
the official HEST-bench per-cancer-type folds). To still get a
non-memorized real-baseline accuracy for this classifier, this module
introduces its own deterministic 80/20 per-WSI split (fixed seed,
`data_utils_classifier.py::build_patch_index_split`) used only for
training/evaluating this classifier — it has no bearing on, and was not
used by, the generative models themselves.

## Usage

```bash
ln -s ~/git/MM-SDG/hpc_experiment_classifier ~/hest-classifier
mkdir -p ~/experiments/hest_classifier

# Train both resolution-matched classifiers
sbatch -J hest_clf_train64 -o ~/experiments/hest_classifier/train64.out -e ~/experiments/hest_classifier/train64.err \
    slurm/job_train_classifier.sh --image-size 64 --epochs 15
sbatch -J hest_clf_train224 -o ~/experiments/hest_classifier/train224.out -e ~/experiments/hest_classifier/train224.err \
    slurm/job_train_classifier.sh --image-size 224 --epochs 15

# Evaluate against hpc_experiment's synthetic output (64x64)
sbatch -J hest_clf_eval64 -o ~/experiments/hest_classifier/eval64.out -e ~/experiments/hest_classifier/eval64.err \
    slurm/job_eval_classifier.sh \
    --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_64.pt \
    --samples-dir /opt/storage/users/$USER/experiments/hest-cdm/synthetic \
    --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_cdm_64.json

# Evaluate against hpc_experiment_sr's cascaded synthetic output (224x224)
sbatch -J hest_clf_eval224 -o ~/experiments/hest_classifier/eval224.out -e ~/experiments/hest_classifier/eval224.err \
    slurm/job_eval_classifier.sh \
    --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_224.pt \
    --samples-dir /opt/storage/users/$USER/experiments/hest-sr/synthetic \
    --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_sr_224.json

# Evaluate the SAME classifiers against the RNA-conditioned experiments'
# paired samples (flat real/synthetic, all CCRCC) -- directly comparable
# to the CCRCC entry above, despite the totally different conditioning
# mechanism (no class label given at all, only real gene expression).
sbatch -J hest_clf_eval_rna64 -o ~/experiments/hest_classifier/eval_rna64.out -e ~/experiments/hest_classifier/eval_rna64.err \
    slurm/job_eval_classifier_on_rna.sh \
    --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_64.pt \
    --samples-dir /opt/storage/users/$USER/experiments/hest-rna/samples \
    --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_rna_64.json

sbatch -J hest_clf_eval_rna_sr224 -o ~/experiments/hest_classifier/eval_rna_sr224.out -e ~/experiments/hest_classifier/eval_rna_sr224.err \
    slurm/job_eval_classifier_on_rna.sh \
    --classifier-checkpoint /opt/storage/users/$USER/experiments/hest-classifier/classifier_224.pt \
    --samples-dir /opt/storage/users/$USER/experiments/hest-rna-sr/samples \
    --output-json /opt/storage/users/$USER/experiments/hest-classifier/eval_rna_sr_224.json
```

Results land in `/opt/storage/users/${USER}/experiments/hest-classifier/`.

## What `eval_classifier.py` reports

- `real_baseline_accuracy`: classifier accuracy on held-out real test
  patches (the 80/20 split above) — upper bound.
- `synthetic_accuracy` / `synthetic_macro_f1` / `synthetic_per_class_accuracy`:
  classifier accuracy on the synthetic images, checking the predicted
  class against the folder's (= conditioning) label — **the actual
  class-conditioning fidelity number**.

`eval_classifier_on_rna.py` reports the same thing in a different shape
(`real_accuracy` / `synthetic_accuracy` against a single fixed
`--true-cancer-type`, since the RNA experiments' samples aren't split into
per-cancer-type folders) for the RNA-conditioned experiments' samples.

## Result

| | real baseline / sanity | **synthetic accuracy (CCRCC)** | overall synthetic accuracy (all 10 classes) |
|---|---|---|---|
| `hpc_experiment` (64×64, class-conditioned) | 0.9184 | **0.2840** | 0.3075 |
| `hpc_experiment_sr` (224×224 cascade, class-conditioned) | 0.9302 | **0.2500** | 0.1690 |
| `hpc_experiment_rna` (64×64, RNA-conditioned, CCRCC only) | 0.8885 (sanity) | **0.3110** | — (CCRCC-only) |
| `hpc_experiment_rna_sr` (224×224 cascade, RNA-conditioned, CCRCC only) | 1.0 (sanity) | **0.2585** | — (CCRCC-only) |

The bottom two rows vs. the top two are the most direct comparison in the
whole thesis: **the same classifier**, same metric, same scale, applied to
two completely different conditioning mechanisms (an explicit discrete
label vs. an implicit real gene-expression embedding that never sees a
cancer-type label at all). RNA-conditioning's CCRCC accuracy is actually
**slightly higher** than class-conditioning's at both resolutions (0.311
vs 0.284 at 64×64; 0.2585 vs 0.2500 at 224×224) — real molecular signal
alone reconstructs recognizable tissue identity at least as well as being
told the label outright, despite never seeing it. This agrees with the
FID and `hpc_experiment_regressor` results: three independent metrics all
point the same direction for RNA-conditioning on CCRCC.

Also notable: CCRCC is consistently the hardest class for the
class-conditioned model too (0.284/0.250 vs. e.g. PRAD's 0.884 and
LYMPH_IDC's 0.767 at 64×64) -- consistent with CCRCC having the worst
per-class FID across every experiment in this repo. The RNA-vs-class
comparison above is therefore on fair, equally-hard ground on both sides.

Read alongside `hpc_experiment_regressor`'s results, this completes the
conditioning-fidelity picture for both strategies: FID says which is more
"realistic", the regressor/classifier pair says which actually *encodes
its condition* faithfully -- and whether that fidelity survives the
cascade to 224×224.
