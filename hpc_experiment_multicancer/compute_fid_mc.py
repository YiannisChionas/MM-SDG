"""Compute FID between the paired real/ and synthetic/ folders produced by
sample_rna_mc.py (held-out WSIs, pooled across the 5 multi-cancer types).

Not an attempt to match the paper's FID50k protocol (10,000 tiles/cancer
type, 50,000 total) -- that's compute we don't have yet at this training
stage. This is a smaller, consistent (same sample count/weights/sampler
settings each time) protocol meant for tracking whether successive
checkpoints are actually improving, not for a literal paper comparison.

Requires `pytorch-fid`. Pre-warm its InceptionV3 weight cache from an
interactive CPU session first if compute nodes lack internet -- see
hpc_experiment/compute_fid.py's docstring for the exact command.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=str, required=True,
                         help="Output dir from sample_rna_mc.py (contains real/ and synthetic/)")
    parser.add_argument("--fid-device", type=str, default="cuda")
    parser.add_argument("--results-json", type=str, default="fid_mc_results.json")
    parser.add_argument("--label", type=str, default="multicancer",
                         help="Key under which to store this run's FID in --results-json, e.g. "
                              "'step1000_raw' -- lets multiple checkpoint evaluations accumulate "
                              "in the same file for an over-training-progress comparison")
    args = parser.parse_args()

    from pytorch_fid.fid_score import calculate_fid_given_paths

    real_path = f"{args.samples_dir}/real"
    synth_path = f"{args.samples_dir}/synthetic"

    fid = calculate_fid_given_paths([real_path, synth_path], batch_size=50, device=args.fid_device, dims=2048)
    print(f"FID [{args.label}]: {fid:.3f}", flush=True)

    try:
        with open(args.results_json) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    results[args.label] = fid
    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.results_json}", flush=True)


if __name__ == "__main__":
    main()
