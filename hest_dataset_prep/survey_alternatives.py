"""One-off: for ALL Visium oncotree codes (not just our current 5), show
sample count + pixel_size_um_estimated distribution, to look for better
replacements for EPM (mixed resolution) and LIHB (consistently coarse,
~0.89-0.92um/px -> always needs upsampling to a 256px/128um patch) --
ideally a code with a comparable sample count (~8-14) but resolution
fine enough (<=0.5um/px) to need no upsampling."""

import argparse

import pandas as pd

TARGET_PATCH_UM = 128.0
TARGET_PX = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    visium = df[df["st_technology"] == "Visium"].copy()

    rows = []
    for code, sub in visium.groupby("oncotree_code"):
        if len(sub) < args.min_samples:
            continue
        px = sub["pixel_size_um_estimated"].dropna()
        if len(px) == 0:
            continue
        crop_px = TARGET_PATCH_UM / px
        n_needs_upsample = int((crop_px < TARGET_PX).sum())
        rows.append({
            "oncotree_code": code,
            "n_samples": len(sub),
            "px_size_min": px.min(),
            "px_size_median": px.median(),
            "px_size_max": px.max(),
            "n_needs_upsample": n_needs_upsample,
            "frac_needs_upsample": n_needs_upsample / len(px),
            "organ": sub["organ"].mode().iloc[0] if not sub["organ"].mode().empty else "?",
        })

    out = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
