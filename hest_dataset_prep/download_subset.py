"""Step 3: download only the files for the selected WSIs (selected_wsis.csv,
produced by select_wsis.py) from the full MahmoodLab/hest HuggingFace
dataset -- NOT the whole ~1TB+ dataset. Follows the official HEST README's
own pattern (filter by id, build one glob pattern per id, snapshot_download
with allow_patterns).

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
    print(f"Downloading files for {len(ids)} selected WSIs: {ids}", flush=True)

    patterns = [f"*{wsi_id}[_.]**" for wsi_id in ids]
    # Always include the metadata CSV itself, cheap and useful to have alongside.
    patterns.append("HEST_v1_3_0.csv")

    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=patterns, local_dir=args.output_dir,
    )
    print(f"Download complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
