"""One-off analysis: which oncotree_code cancer types have the most
Visium-specifically samples in the HEST-1k catalog, to pick 5 realistic
cancer types for a paper-scale reproduction attempt (the paper's own
LUAD/KIRP/CESC/COAD/GBM doesn't hold up against the real catalog -- KIRP
has 0 samples, CESC has 1, GBM has 4)."""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--min-samples", type=int, default=8,
                         help="Minimum Visium samples to consider a cancer type usable")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    visium = df[df["st_technology"] == "Visium"]
    print(f"Total Visium samples: {len(visium)} / {len(df)}", flush=True)

    counts = visium["oncotree_code"].value_counts()
    print("\n--- oncotree_code counts, Visium only (sorted) ---", flush=True)
    print(counts, flush=True)

    usable = counts[counts >= args.min_samples]
    print(f"\n--- Cancer types with >= {args.min_samples} Visium samples ---", flush=True)
    print(usable, flush=True)

    # Cross-check: how many unique patients per code (not just samples/sections),
    # to flag codes where one patient dominates (risk of leakage/low diversity).
    print("\n--- unique patients per code (Visium, for codes above threshold) ---", flush=True)
    for code in usable.index:
        sub = visium[visium["oncotree_code"] == code]
        n_patients = sub["patient"].nunique() if "patient" in sub.columns else "N/A"
        print(f"{code}: {len(sub)} samples, {n_patients} unique patients", flush=True)


if __name__ == "__main__":
    main()
