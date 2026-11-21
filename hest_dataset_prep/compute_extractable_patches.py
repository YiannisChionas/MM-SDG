"""Step 4: for each of the 108 candidate WSIs (now that we have their
st/*.h5ad spot-coordinate files locally), compute how many of its
in-tissue spots can actually be extracted as a patch of FIXED PHYSICAL
SIZE (default 128um, matching the paper's 256px @ 0.5um/px) -- i.e. how
many spots have a crop window of crop_px = round(128um / pixel_size_um)
that fits entirely inside the WSI's full-resolution pixel bounds.

This replaces the raw `spots_under_tissue` metadata field (which is
agnostic to patch size) with an extraction-aware count: at a fixed
physical patch size, high-resolution WSIs (small pixel_size_um, e.g.
COAD's ~0.276) need a LARGER raw-pixel crop window, so spots near the
tissue/image edge are more likely to be excluded than for low-resolution
WSIs -- shifting the effective per-type patch
budget compared to what spots_under_tissue alone would suggest.

No image data is touched -- only spot pixel coordinates (from .h5ad)
and each WSI's full-resolution dimensions (from the metadata CSV).

Also excludes WSIs whose .h5ad has fewer than --min-genes genes
(default 15,000): discovered that one COAD WSI (TENX28) has only 1,142
genes -- a targeted/custom probe panel, not full transcriptome -- which
on its own collapsed the 5-cancer-type gene intersection from ~17,311 to
1,064 in select_genes_multicancer.py. A single low-coverage WSI pulled
into the candidate pool is enough to wreck the downstream gene panel,
so this filters them out before they ever reach WSI selection.
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd

TARGET_CODES = ["COAD", "SCCRCC", "EPM", "PRAD", "IDC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--adata-dir", type=str, required=True, help="Directory containing st/*.h5ad")
    parser.add_argument("--patch-size-um", type=float, default=128.0,
                         help="Target physical patch size in micrometers (256px @ 0.5um/px = 128um)")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--min-genes", type=int, default=15000,
                         help="Exclude WSIs whose .h5ad has fewer genes than this (targeted/custom "
                              "panels, not full transcriptome) -- see module docstring for why")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    # species filter is a hard safety check -- LIHB was discovered to be
    # entirely Mus musculus despite looking like a normal human cancer
    # code, which silently produced a zero-gene-overlap pool downstream.
    visium = df[
        (df["st_technology"] == "Visium") & (df["oncotree_code"].isin(TARGET_CODES)) &
        (df["species"] == "Homo sapiens")
    ].copy()
    print(f"Candidate Visium WSIs (Homo sapiens only): {len(visium)}", flush=True)

    results = []
    for _, row in visium.iterrows():
        wsi_id = row["id"]
        px_size = row["pixel_size_um_estimated"]
        width, height = row["fullres_px_width"], row["fullres_px_height"]
        if pd.isna(px_size) or pd.isna(width) or pd.isna(height):
            print(f"[warn] {wsi_id}: missing pixel_size/dimensions, skipping", flush=True)
            continue

        adata_path = os.path.join(args.adata_dir, "st", f"{wsi_id}.h5ad")
        if not os.path.exists(adata_path):
            print(f"[warn] {wsi_id}: missing {adata_path}, skipping", flush=True)
            continue

        a = ad.read_h5ad(adata_path)
        if a.n_vars < args.min_genes:
            print(f"[warn] {wsi_id} ({row['oncotree_code']}): only {a.n_vars} genes "
                  f"(< --min-genes {args.min_genes}), likely a targeted panel -- excluding", flush=True)
            continue

        in_tissue = a.obs["in_tissue"].to_numpy().astype(bool) if "in_tissue" in a.obs.columns else np.ones(a.n_obs, bool)
        rows_px = a.obs["pxl_row_in_fullres"].to_numpy()[in_tissue]
        cols_px = a.obs["pxl_col_in_fullres"].to_numpy()[in_tissue]
        raw_spots = int(in_tissue.sum())

        crop_px = int(round(args.patch_size_um / px_size))
        half = crop_px / 2.0
        fits = (
            (rows_px - half >= 0) & (rows_px + half <= height) &
            (cols_px - half >= 0) & (cols_px + half <= width)
        )
        extractable = int(fits.sum())

        print(f"{wsi_id} ({row['oncotree_code']}): px_size={px_size:.3f}um crop_px={crop_px} "
              f"raw_spots={raw_spots} extractable={extractable} "
              f"({100*extractable/max(raw_spots,1):.1f}% kept)", flush=True)

        results.append({
            "id": wsi_id, "oncotree_code": row["oncotree_code"],
            "patient": row.get("patient"), "n_vars": a.n_vars,
            "pixel_size_um_estimated": px_size, "crop_px": crop_px,
            "raw_spots_under_tissue": raw_spots, "extractable_spots": extractable,
        })

    out = pd.DataFrame(results)
    out.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(out)} rows to {args.output_csv}", flush=True)

    print("\n--- Per-type totals (extractable_spots) ---", flush=True)
    print(out.groupby("oncotree_code")["extractable_spots"].agg(["count", "sum", "mean"]), flush=True)
    print("\n--- Per-type totals (raw_spots_under_tissue, for comparison) ---", flush=True)
    print(out.groupby("oncotree_code")["raw_spots_under_tissue"].agg(["count", "sum", "mean"]), flush=True)


if __name__ == "__main__":
    main()
