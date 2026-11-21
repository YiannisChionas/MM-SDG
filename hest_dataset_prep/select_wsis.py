"""Select up to 10 WSIs per cancer type for our 5 target Visium cancer
types (COAD, SCCRCC, EPM, PRAD, IDC -- PRAD only has 7 human candidates
total, so it takes all of them), chosen to balance total patch count
across types -- not just WSI count.

Default --count-column is now `extractable_spots` (from
compute_extractable_patches.py), which accounts for each WSI's own
pixel_size_um at a FIXED physical patch size -- not the raw
`spots_under_tissue` metadata field, which is patch-size-agnostic and
would overweight low-resolution (coarse pixel_size_um) WSIs once we
extract at a fixed physical size (see compute_extractable_patches.py's
docstring). Pass --count-column spots_under_tissue to reproduce the
original (patch-size-naive) selection.

For types with exactly --n-per-type candidates, all are used. For types
with more, the N WSIs whose count is closest to the overall median
(across all candidate WSIs of all 5 types) are chosen, so each type's
total patch budget ends up similar instead of whichever N happen to be
largest.
"""

import argparse

import pandas as pd

TARGET_CODES = ["COAD", "SCCRCC", "EPM", "PRAD", "IDC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True,
                         help="Either the raw HEST metadata CSV, or compute_extractable_patches.py's output")
    parser.add_argument("--count-column", type=str, default="extractable_spots")
    parser.add_argument("--n-per-type", type=int, default=10)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    if "st_technology" in df.columns:
        df = df[df["st_technology"] == "Visium"]
    visium = df[df["oncotree_code"].isin(TARGET_CODES)].copy()
    print(f"Candidate Visium WSIs across {TARGET_CODES}: {len(visium)}", flush=True)

    col = args.count_column
    overall_median = visium[col].median()
    print(f"Overall median {col} across all candidates: {overall_median:.0f}", flush=True)

    selected_rows = []
    for code in TARGET_CODES:
        sub = visium[visium["oncotree_code"] == code].copy()
        if len(sub) <= args.n_per_type:
            chosen = sub
        else:
            sub["dist_to_median"] = (sub[col] - overall_median).abs()
            chosen = sub.sort_values("dist_to_median").head(args.n_per_type)
        total_patches = chosen[col].sum()
        print(f"{code}: selected {len(chosen)} WSIs, total {col}={total_patches:.0f}, "
              f"mean/WSI={chosen[col].mean():.0f}, "
              f"min={chosen[col].min():.0f}, max={chosen[col].max():.0f}",
              flush=True)
        selected_rows.append(chosen.drop(columns=["dist_to_median"], errors="ignore"))

    result = pd.concat(selected_rows, ignore_index=True)
    result.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(result)} selected WSIs to {args.output_csv}", flush=True)
    print(f"\nTotal {col} across all 5 types: {result[col].sum():.0f}", flush=True)
    print("\nPer-type totals:")
    print(result.groupby("oncotree_code")[col].agg(["count", "sum", "mean"]), flush=True)


if __name__ == "__main__":
    main()
