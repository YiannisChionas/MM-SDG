"""Run the SAME cancer-type classifier (trained in train_classifier.py,
used in eval_classifier.py to judge the class-conditioned experiments) on
the RNA-conditioned synthetic tiles instead (hpc_experiment_rna's
sample_rna.py / hpc_experiment_rna_sr's sample_rna_sr.py output).

Those experiments never give the model a cancer-type label -- only a real
gene-expression embedding -- but every sample is from CCRCC patients (the
RNA experiments are CCRCC-only). So "does the classifier recognize this as
CCRCC tissue?" is a class-conditioning-fidelity number computed on a
*completely different conditioning mechanism*, directly comparable to the
CCRCC entry of eval_classifier.py's `synthetic_per_class_accuracy` for the
class-conditioned experiments -- same classifier, same metric, same scale,
no FID resolution/scope caveats.

Expects --samples-dir with FLAT real/ and synthetic/ subfolders (the
sample_rna.py / sample_rna_sr.py paired-sampling layout), not the
per-cancer-type subfolders that sample.py / sample_sr.py produce.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval_classifier import load_classifier


def eval_flat_folder(model, image_size, folder, device, batch_size):
    paths = sorted(Path(folder).glob("*.png"))
    preds = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
                t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
                imgs.append(t)
            x = torch.stack(imgs).to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--samples-dir", type=str, required=True,
                         help="Output dir of sample_rna.py / sample_rna_sr.py (flat real/ and synthetic/, all CCRCC)")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--true-cancer-type", type=str, default="CCRCC")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    model, cancer_types, image_size = load_classifier(args.classifier_checkpoint, device)
    true_idx = cancer_types.index(args.true_cancer_type)
    print(f"Loaded classifier (image_size={image_size}); true class = {args.true_cancer_type} (idx {true_idx})",
          flush=True)

    samples_dir = Path(args.samples_dir)
    real_preds = eval_flat_folder(model, image_size, samples_dir / "real", device, args.batch_size)
    synth_preds = eval_flat_folder(model, image_size, samples_dir / "synthetic", device, args.batch_size)
    print(f"Evaluated {len(real_preds)} real / {len(synth_preds)} synthetic images", flush=True)

    real_acc = float(np.mean(np.array(real_preds) == true_idx)) if real_preds else None
    synth_acc = float(np.mean(np.array(synth_preds) == true_idx)) if synth_preds else None
    print(f"Real accuracy (sanity check): {real_acc}", flush=True)
    print(f"Synthetic accuracy (RNA-conditioning's implicit class fidelity): {synth_acc}", flush=True)

    results = {
        "image_size": image_size,
        "true_cancer_type": args.true_cancer_type,
        "n_real": len(real_preds),
        "n_synthetic": len(synth_preds),
        "real_accuracy": real_acc,
        "synthetic_accuracy": synth_acc,
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
