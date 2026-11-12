"""Export a matching real-patch sample and compute FID against synthetic images.

Computes FID both per cancer type and pooled across all classes.

Requires `pytorch-fid` (pip install pytorch-fid). Its InceptionV3 weights are
downloaded from the internet on first use -- compute nodes on this cluster
may not have internet access, so pre-warm the cache once from an
interactive CPU session before submitting the GPU job:

    srun --partition=yoda --nodes=1 --ntasks-per-node=1 --gpus-per-task=0 \
         --time=00:15:00 --pty bash -i
    conda activate sdg
    python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3])"
"""

import argparse
import glob
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import h5py
from PIL import Image

from data_utils import CANCER_TYPES


def export_real_samples(hest_dir, out_dir, per_class, image_size, seed=0):
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    for cancer_type in CANCER_TYPES:
        class_dir = out_dir / cancer_type
        class_dir.mkdir(parents=True, exist_ok=True)
        if len(list(class_dir.glob("*.png"))) >= per_class:
            print(f"{cancer_type}: real sample already exported, skipping", flush=True)
            continue

        files = sorted(glob.glob(os.path.join(hest_dir, cancer_type, "patches", "*.h5")))
        pool = []
        for fp in files:
            with h5py.File(fp, "r") as f:
                n = f["img"].shape[0]
            pool.extend((fp, i) for i in range(n))

        if len(pool) < per_class:
            print(f"[warn] {cancer_type}: only {len(pool)} real patches available, "
                  f"requested {per_class} -- using all of them", flush=True)

        rng.shuffle(pool)
        chosen = pool[:per_class]
        print(f"{cancer_type}: exporting {len(chosen)} real patches (pool size {len(pool)})", flush=True)

        by_file = {}
        for fp, i in chosen:
            by_file.setdefault(fp, []).append(i)

        idx_counter = 0
        for fp, idxs in by_file.items():
            with h5py.File(fp, "r") as f:
                for i in sorted(idxs):
                    img = f["img"][i]
                    im = Image.fromarray(img).resize((image_size, image_size), Image.BILINEAR)
                    im.save(class_dir / f"{idx_counter:05d}.png")
                    idx_counter += 1


def pool_class_dirs(root_dirs_and_dest):
    """Flatten per-class subfolders into one directory with class-prefixed
    filenames, for pooled (all-classes-together) FID."""
    for root, dest in root_dirs_and_dest:
        dest.mkdir(parents=True, exist_ok=True)
        for cancer_type in CANCER_TYPES:
            class_dir = Path(root) / cancer_type
            if not class_dir.is_dir():
                continue
            for f in class_dir.glob("*.png"):
                shutil.copy(f, dest / f"{cancer_type}_{f.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-dir", type=str, required=True,
                         help="Staged hest-bench directory to sample real patches from")
    parser.add_argument("--synthetic-dir", type=str, required=True,
                         help="Output directory from sample.py (contains <CANCER_TYPE>/*.png)")
    parser.add_argument("--real-dir", type=str, required=True,
                         help="Where to write the exported real-patch sample")
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--fid-device", type=str, default="cuda")
    parser.add_argument("--results-json", type=str, default="fid_results.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    export_real_samples(args.hest_dir, args.real_dir, args.per_class, args.image_size, seed=args.seed)

    from pytorch_fid.fid_score import calculate_fid_given_paths

    results = {}

    for cancer_type in CANCER_TYPES:
        real_path = os.path.join(args.real_dir, cancer_type)
        synth_path = os.path.join(args.synthetic_dir, cancer_type)
        if not os.path.isdir(synth_path) or not os.listdir(synth_path):
            print(f"Skipping {cancer_type}: no synthetic images found at {synth_path}", flush=True)
            continue
        fid = calculate_fid_given_paths(
            [real_path, synth_path], batch_size=50, device=args.fid_device, dims=2048,
        )
        results[cancer_type] = fid
        print(f"FID [{cancer_type}]: {fid:.3f}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        pooled_real = Path(tmp) / "real"
        pooled_synth = Path(tmp) / "synth"
        pool_class_dirs([(args.real_dir, pooled_real), (args.synthetic_dir, pooled_synth)])
        pooled_fid = calculate_fid_given_paths(
            [str(pooled_real), str(pooled_synth)], batch_size=50, device=args.fid_device, dims=2048,
        )
        results["POOLED"] = pooled_fid
        print(f"FID [POOLED]: {pooled_fid:.3f}", flush=True)

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.results_json}", flush=True)


if __name__ == "__main__":
    main()
