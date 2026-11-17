"""Evaluate a trained image->cancer-type classifier (train_classifier.py) as
a judge of class-conditioning fidelity -- the discrete-label analogue of
hpc_experiment_regressor's image->expression regressor:

1. Real baseline accuracy: classifier on held-out-WSI real test patches
   (same deterministic 80/20 split used at training time) -- upper bound on
   how well cancer type can be predicted from a real image at all.
2. Synthetic accuracy: given a --samples-dir produced by
   hpc_experiment/sample.py or hpc_experiment_sr/sample_sr.py
   (per-cancer-type subfolders of .png files), run the classifier on every
   image and check whether the predicted class matches the folder's
   (= conditioning) label. Reports overall accuracy, macro-F1, and
   per-class accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_utils_classifier import (
    CANCER_TYPES, build_patch_index_split, HestPatchDataset, ImageToClassClassifier,
)


def load_classifier(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ImageToClassClassifier(num_classes=len(ckpt["cancer_types"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["cancer_types"], ckpt["image_size"]


def eval_real_baseline(model, image_size, data_dir, device, seed, batch_size):
    _, test_index = build_patch_index_split(data_dir, CANCER_TYPES, test_fraction=0.2, seed=seed)
    print(f"Real baseline: {len(test_index)} held-out test patches", flush=True)
    dataset = HestPatchDataset(test_index, image_size=image_size)
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            x = torch.stack([b[0] for b in batch]).to(device)
            y = torch.tensor([b[1] for b in batch])
            pred = model(x).argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total


def eval_samples_dir(model, cancer_types, image_size, samples_dir, device, batch_size, max_per_class=None):
    samples_dir = Path(samples_dir)
    all_true, all_pred = [], []
    per_class_acc = {}
    for class_idx, ct in enumerate(cancer_types):
        class_dir = samples_dir / ct
        paths = sorted(class_dir.glob("*.png"))
        if max_per_class is not None:
            paths = paths[:max_per_class]
        if not paths:
            print(f"[warn] no images found for {ct} under {class_dir}")
            continue

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

        true = [class_idx] * len(preds)
        all_true.extend(true)
        all_pred.extend(preds)
        acc = float(np.mean(np.array(preds) == class_idx))
        per_class_acc[ct] = acc
        print(f"  {ct}: accuracy {acc:.4f} ({len(preds)} images)", flush=True)

    overall_acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
    try:
        from sklearn.metrics import f1_score
        macro_f1 = float(f1_score(all_true, all_pred, average="macro", labels=list(range(len(cancer_types)))))
    except ImportError:
        macro_f1 = None

    return overall_acc, macro_f1, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--samples-dir", type=str, required=True,
                         help="Output dir of sample.py / sample_sr.py (per-cancer-type subfolders)")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    model, cancer_types, image_size = load_classifier(args.classifier_checkpoint, device)
    print(f"Loaded classifier (image_size={image_size}, {len(cancer_types)} classes)", flush=True)

    real_baseline_acc = eval_real_baseline(model, image_size, args.data_dir, device, args.seed, args.batch_size)
    print(f"Real baseline accuracy: {real_baseline_acc:.4f}", flush=True)

    synth_acc, synth_f1, per_class_acc = eval_samples_dir(
        model, cancer_types, image_size, args.samples_dir, device, args.batch_size, args.max_per_class,
    )
    print(f"Synthetic accuracy: {synth_acc:.4f} (macro-F1: {synth_f1})", flush=True)

    results = {
        "image_size": image_size,
        "n_classes": len(cancer_types),
        "real_baseline_accuracy": real_baseline_acc,
        "synthetic_accuracy": synth_acc,
        "synthetic_macro_f1": synth_f1,
        "synthetic_per_class_accuracy": per_class_acc,
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
