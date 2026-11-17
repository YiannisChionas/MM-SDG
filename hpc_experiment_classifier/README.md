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
```

Results land in `/opt/storage/users/${USER}/experiments/hest-classifier/`.

## What `eval_classifier.py` reports

- `real_baseline_accuracy`: classifier accuracy on held-out real test
  patches (the 80/20 split above) — upper bound.
- `synthetic_accuracy` / `synthetic_macro_f1` / `synthetic_per_class_accuracy`:
  classifier accuracy on the synthetic images, checking the predicted
  class against the folder's (= conditioning) label — **the actual
  class-conditioning fidelity number**.

## Result

*(to be filled in once both classifiers are trained and evaluated)*

| | real_baseline_accuracy | synthetic_accuracy | synthetic_macro_f1 |
|---|---|---|---|
| `hpc_experiment` (64×64) | | | |
| `hpc_experiment_sr` (224×224 cascade) | | | |

Read alongside `hpc_experiment_regressor`'s results, this completes the
conditioning-fidelity picture for both strategies: FID says which is more
"realistic", this (and the regressor) say which actually *encodes its
condition* faithfully -- and whether that fidelity survives the cascade
to 224×224.
