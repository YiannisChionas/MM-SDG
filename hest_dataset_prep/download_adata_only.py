"""Step 2.5: download ONLY the .h5ad spot-coordinate/expression files
(st/*.h5ad) for all human (Homo sapiens) Visium candidate WSIs across
our 5 target cancer types (COAD, SCCRCC, EPM, PRAD, IDC) -- not just the
WSIs already selected by select_wsis.py. Needed to compute the
extraction-aware patch count per WSI (how many spots actually fit a
fixed-physical-size crop window, given each WSI's own pixel_size_um)
before finalizing which WSIs per type to keep and before downloading
any WSI images.

species=='Homo sapiens' is a REQUIRED filter here: LIHB was originally
in this list, but turned out to be 100% Mus musculus (mouse) despite
looking like a normal human cancer oncotree code -- discovered only
because the downstream gene-panel intersection across all 5 types came
out to zero. Replaced with PRAD.
"""

import argparse

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "MahmoodLab/hest"
TARGET_CODES = ["COAD", "SCCRCC", "EPM", "PRAD", "IDC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    visium = df[
        (df["st_technology"] == "Visium") & (df["oncotree_code"].isin(TARGET_CODES)) &
        (df["species"] == "Homo sapiens")
    ]
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
