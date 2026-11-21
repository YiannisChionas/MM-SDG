"""Select 10 WSIs per cancer type for our 5 target Visium cancer types
(the top 5 by Visium sample count: COAD, SCCRCC, EPM, LIHB, IDC), chosen
to balance total patch/spot count across types -- not just WSI count.

For types with exactly 10 Visium samples (LIHB, IDC), all 10 are used.
For types with more (COAD, SCCRCC, EPM), the 10 WSIs whose
spots_under_tissue is closest to the overall median (across all
candidate WSIs of all 5 types) are chosen, so each type's total patch
budget ends up similar instead of whichever 10 happen to be largest.
"""

import argparse

import numpy as np
import pandas as pd

TARGET_CODES = ["COAD", "SCCRCC", "EPM", "LIHB", "IDC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--n-per-type", type=int, default=10)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    visium = df[(df["st_technology"] == "Visium") & (df["oncotree_code"].isin(TARGET_CODES))].copy()
    print(f"Candidate Visium WSIs across {TARGET_CODES}: {len(visium)}", flush=True)

    overall_median = visium["spots_under_tissue"].median()
    print(f"Overall median spots_under_tissue across all candidates: {overall_median:.0f}", flush=True)

    selected_rows = []
    for code in TARGET_CODES:
        sub = visium[visium["oncotree_code"] == code].copy()
        if len(sub) <= args.n_per_type:
            chosen = sub
        else:
            sub["dist_to_median"] = (sub["spots_under_tissue"] - overall_median).abs()
            chosen = sub.sort_values("dist_to_median").head(args.n_per_type)
        total_patches = chosen["spots_under_tissue"].sum()
        print(f"{code}: selected {len(chosen)} WSIs, total patches={total_patches:.0f}, "
              f"mean patches/WSI={chosen['spots_under_tissue'].mean():.0f}, "
              f"min={chosen['spots_under_tissue'].min():.0f}, max={chosen['spots_under_tissue'].max():.0f}",
              flush=True)
        selected_rows.append(chosen.drop(columns=["dist_to_median"], errors="ignore"))

    result = pd.concat(selected_rows, ignore_index=True)
    result.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(result)} selected WSIs to {args.output_csv}", flush=True)
    print(f"\nTotal patches across all 5 types: {result['spots_under_tissue'].sum():.0f}", flush=True)
    print("\nPer-type totals:")
    print(result.groupby("oncotree_code")["spots_under_tissue"].agg(["count", "sum", "mean"]), flush=True)


if __name__ == "__main__":
    main()
