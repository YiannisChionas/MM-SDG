"""Compute FID between the paired real/ and synthetic/ folders produced by
sample_rna_sr.py (held-out WSIs, cascaded RNA-expression-conditioned
generation at native 224x224).

Requires `pytorch-fid`. InceptionV3 weights already pre-warmed in
~/.cache/torch/hub/checkpoints/ from the earlier experiments.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=str, required=True,
                         help="Output dir from sample_rna_sr.py (contains real/ and synthetic/)")
    parser.add_argument("--fid-device", type=str, default="cuda")
    parser.add_argument("--results-json", type=str, default="fid_rna_sr_results.json")
    args = parser.parse_args()

    from pytorch_fid.fid_score import calculate_fid_given_paths

    real_path = f"{args.samples_dir}/real"
    synth_path = f"{args.samples_dir}/synthetic"

    fid = calculate_fid_given_paths([real_path, synth_path], batch_size=50, device=args.fid_device, dims=2048)
    print(f"FID [CCRCC, RNA-conditioned cascaded SR, held-out WSIs, 224x224]: {fid:.3f}", flush=True)

    with open(args.results_json, "w") as f:
        json.dump({"CCRCC_rna_conditioned_sr_224": fid}, f, indent=2)
    print(f"Saved results to {args.results_json}", flush=True)


if __name__ == "__main__":
    main()
