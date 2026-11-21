"""Step 2.5: download ONLY the .h5ad spot-coordinate/expression files
(st/*.h5ad, ~12.5GB total, not the ~242GB of WSI images/patches/etc.)
for all 108 Visium candidate WSIs across our 5 target cancer types
(COAD, SCCRCC, EPM, LIHB, IDC) -- not just the 50 already selected by
select_wsis.py. Needed to compute the extraction-aware patch count per
WSI (how many spots actually fit a fixed-physical-size crop window,
given each WSI's own pixel_size_um) before finalizing which 10 WSIs per
type to keep and before downloading any WSI images.
"""

import argparse

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "MahmoodLab/hest"
TARGET_CODES = ["COAD", "SCCRCC", "EPM", "LIHB", "IDC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    visium = df[(df["st_technology"] == "Visium") & (df["oncotree_code"].isin(TARGET_CODES))]
    ids = visium["id"].tolist()
    print(f"Downloading st/*.h5ad for {len(ids)} candidate WSIs", flush=True)

    patterns = [f"st/{wsi_id}.h5ad" for wsi_id in ids]

    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=patterns, local_dir=args.output_dir,
    )
    print(f"Download complete: {args.output_dir}/st/", flush=True)


if __name__ == "__main__":
    main()
