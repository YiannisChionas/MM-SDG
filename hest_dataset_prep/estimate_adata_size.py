"""One-off: estimate total download size of just the .h5ad (spot
coordinates + expression) files for our 108 candidate WSIs, WITHOUT
downloading anything -- uses HfApi to list repo files and their sizes
remotely, to decide if fetching all of them for the extraction-aware
spot-count recalculation is cheap enough to just do."""

import argparse

import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "MahmoodLab/hest"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    target_codes = ["COAD", "SCCRCC", "EPM", "LIHB", "IDC"]
    visium = df[(df["st_technology"] == "Visium") & (df["oncotree_code"].isin(target_codes))]
    ids = set(visium["id"].tolist())
    print(f"Candidate WSIs: {len(ids)}", flush=True)

    api = HfApi()
    print("Listing all repo files with sizes (this may take a moment)...", flush=True)
    files_info = list(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset"))
    print(f"Total files in repo: {len(files_info)}", flush=True)

    # Find adata/expression files belonging to our candidate ids
    matched = [f for f in files_info if any(f"{i}." in f or f"{i}_" in f for i in ids)]
    print(f"\nSample matched filenames (first 20):")
    for f in matched[:20]:
        print(" ", f)

    # Get sizes via get_paths_info (batched)
    print("\nFetching sizes for matched files...", flush=True)
    infos = api.get_paths_info(repo_id=REPO_ID, repo_type="dataset", paths=matched)
    total_bytes = sum(i.size for i in infos if i.size)
    by_ext = {}
    for i in infos:
        if not i.size:
            continue
        ext = i.path.rsplit(".", 1)[-1] if "." in i.path else "noext"
        by_ext[ext] = by_ext.get(ext, 0) + i.size

    print(f"\nTotal size of ALL matched files (incl. WSIs/images): {total_bytes / 1e9:.2f} GB", flush=True)
    print("By extension:")
    for ext, sz in sorted(by_ext.items(), key=lambda x: -x[1]):
        print(f"  .{ext}: {sz / 1e9:.3f} GB")

    adata_bytes = sum(i.size for i in infos if i.size and ("h5ad" in i.path))
    print(f"\nTotal size of .h5ad files only: {adata_bytes / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
