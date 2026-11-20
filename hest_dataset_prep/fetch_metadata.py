"""Step 1 of downloading a USEFUL SUBSET of the full HEST-1k dataset
(MahmoodLab/hest on HuggingFace, ~1TB+ total -- NOT the small
hest-bench benchmark subset we already have locally) to the cluster's
shared /opt/storage/datasets.

Only fetches the metadata CSV (a few MB), NOT any actual WSI/expression
data. Saves a local copy and prints the available oncotree_code/organ/
st_technology values so we can confirm the exact filter to use for our 5
target cancer types (LUAD, KIRP, CESC, COAD, GBM) before committing to
downloading any real data -- the shared cluster storage only has ~1.1TB
free, so we do NOT want to guess wrong and download the wrong subset.
"""

import argparse

import pandas as pd

HEST_CSV_URL = "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    print(f"Reading metadata from {HEST_CSV_URL} ...", flush=True)
    meta_df = pd.read_csv(HEST_CSV_URL)
    print(f"Total samples in catalog: {len(meta_df)}", flush=True)
    print(f"Columns: {list(meta_df.columns)}", flush=True)

    meta_df.to_csv(args.output_csv, index=False)
    print(f"Saved full metadata catalog to {args.output_csv}", flush=True)

    if "st_technology" in meta_df.columns:
        print("\n--- st_technology value counts ---", flush=True)
        print(meta_df["st_technology"].value_counts(), flush=True)

    if "oncotree_code" in meta_df.columns:
        print("\n--- oncotree_code value counts (top 40) ---", flush=True)
        print(meta_df["oncotree_code"].value_counts().head(40), flush=True)

    if "organ" in meta_df.columns:
        print("\n--- organ value counts ---", flush=True)
        print(meta_df["organ"].value_counts(), flush=True)

    # Best-effort guess at matches for our 5 target cancer types -- to be
    # manually confirmed against the full value_counts above before
    # committing to a real download, not used directly.
    candidates = {
        "LUAD": ["LUAD"],
        "KIRP": ["KIRP"],
        "CESC": ["CESC"],
        "COAD": ["COAD"],
        "GBM": ["GBM"],
    }
    if "oncotree_code" in meta_df.columns:
        print("\n--- naive oncotree_code matches for target cancer types ---", flush=True)
        for label, codes in candidates.items():
            match = meta_df[meta_df["oncotree_code"].isin(codes)]
            print(f"{label}: {len(match)} samples matching {codes}", flush=True)


if __name__ == "__main__":
    main()
