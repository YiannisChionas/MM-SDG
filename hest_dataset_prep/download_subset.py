"""Step 5: download the full-resolution WSI images (wsis/{id}.tif, the
heavy part -- ~79GB estimated for our 50 selected WSIs, scaled from the
~170.8GB/108-WSI estimate) for the final selected_wsis.csv produced by
select_wsis.py. We already have st/*.h5ad for all 108 candidates
(download_adata_only.py) -- this step only needs to fetch the WSI tiffs
themselves, not the cellvit_seg/parquet/thumbnails/etc. files that
matching every file for an id would also pull in.

Resumable: snapshot_download skips files already present locally, so
re-running this (e.g. after a job hits the time limit) picks up where it
left off.
"""

import argparse

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "MahmoodLab/hest"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.selected_wsis_csv)
    ids = df["id"].tolist()
    print(f"Downloading wsis/*.tif for {len(ids)} selected WSIs: {ids}", flush=True)

    patterns = [f"wsis/{wsi_id}.tif" for wsi_id in ids]

    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=patterns, local_dir=args.output_dir,
    )
    print(f"Download complete: {args.output_dir}/wsis/", flush=True)


if __name__ == "__main__":
    main()
