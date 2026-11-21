"""Custom patch extraction for the multi-cancer paper-scale reproduction
(see DECISIONS.md). For each of the 50 selected WSIs (selected_wsis.csv,
produced in hest_dataset_prep/), crops a FIXED PHYSICAL SIZE window
(default 128um, matching the paper's 256px@0.5um/px) around every
in-tissue Visium spot's pixel coordinates, resizes it to 256x256, and
saves the result as patches/{id}.h5 -- same (img, barcode) structure as
HEST's own patches/*.h5 files, so downstream expression-matching code
can barcode-join against that WSI's st/{id}.h5ad exactly like every
other experiment in this repo (no gene panel baked in here).

Applies the paper's "maximum of randomly chosen 4000 tiles per slide"
cap (DECISIONS.md #5) via a fixed-seed random subsample when a WSI has
more than 4000 extractable spots.

Reads WSIs lazily via tifffile+zarr (confirmed to do real tile-level
reads, not full-image loads) -- never loads a whole WSI into memory.
Resumable: skips a WSI if its output .h5 already exists.
"""

import argparse
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image

MAX_TILES_PER_SLIDE = 4000


def crop_and_resize(zarr_arr, row, col, crop_px, out_px=256):
    half = crop_px // 2
    crop = np.asarray(zarr_arr[row - half:row + half, col - half:col + half])
    if crop.shape[0] != crop_px or crop.shape[1] != crop_px:
        # Shouldn't happen for spots already filtered as "extractable", but
        # guard against off-by-one from odd crop_px.
        pad_r, pad_c = crop_px - crop.shape[0], crop_px - crop.shape[1]
        crop = np.pad(crop, ((0, pad_r), (0, pad_c), (0, 0)), mode="edge")
    img = Image.fromarray(crop).resize((out_px, out_px), Image.BILINEAR)
    return np.asarray(img)


def extract_one_wsi(wsi_id, wsis_dir, adata_dir, out_dir, crop_px, out_px, seed):
    out_path = os.path.join(out_dir, f"{wsi_id}.h5")
    if os.path.exists(out_path):
        print(f"{wsi_id}: already extracted, skipping", flush=True)
        return

    adata_path = os.path.join(adata_dir, "st", f"{wsi_id}.h5ad")
    wsi_path = os.path.join(wsis_dir, f"{wsi_id}.tif")
    a = ad.read_h5ad(adata_path)

    in_tissue = a.obs["in_tissue"].to_numpy().astype(bool) if "in_tissue" in a.obs.columns else np.ones(a.n_obs, bool)
    rows_px = a.obs["pxl_row_in_fullres"].to_numpy()[in_tissue]
    cols_px = a.obs["pxl_col_in_fullres"].to_numpy()[in_tissue]
    barcodes = a.obs_names.to_numpy()[in_tissue]

    store = tifffile.imread(wsi_path, aszarr=True, key=0)
    z = zarr.open(store, mode="r")
    height, width = z.shape[0], z.shape[1]

    half = crop_px / 2.0
    fits = (
        (rows_px - half >= 0) & (rows_px + half <= height) &
        (cols_px - half >= 0) & (cols_px + half <= width)
    )
    rows_px, cols_px, barcodes = rows_px[fits], cols_px[fits], barcodes[fits]
    print(f"{wsi_id}: {fits.sum()} extractable spots (of {len(in_tissue)} in-tissue)", flush=True)

    if len(rows_px) > MAX_TILES_PER_SLIDE:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(rows_px), size=MAX_TILES_PER_SLIDE, replace=False)
        idx.sort()
        rows_px, cols_px, barcodes = rows_px[idx], cols_px[idx], barcodes[idx]
        print(f"{wsi_id}: capped to {MAX_TILES_PER_SLIDE} tiles (paper's per-slide max)", flush=True)

    imgs = np.empty((len(rows_px), out_px, out_px, 3), dtype=np.uint8)
    for i, (r, c) in enumerate(zip(rows_px, cols_px)):
        imgs[i] = crop_and_resize(z, int(r), int(c), crop_px, out_px)
        if i % 500 == 0:
            print(f"  {wsi_id}: {i}/{len(rows_px)}", flush=True)

    store.close()

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("img", data=imgs, compression="gzip", compression_opts=4)
        f.create_dataset("barcode", data=np.array(barcodes, dtype="S")[:, None])
    os.replace(tmp_path, out_path)
    print(f"{wsi_id}: saved {len(imgs)} patches to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--hest-dir", type=str, required=True, help="Dir containing wsis/ and st/")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--out-px", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oncotree-code", type=str, default=None,
                         help="If set, only process WSIs of this cancer type (for parallel per-type jobs)")
    args = parser.parse_args()

    df = pd.read_csv(args.selected_wsis_csv)
    if args.oncotree_code:
        df = df[df["oncotree_code"] == args.oncotree_code]
    print(f"Extracting patches for {len(df)} WSIs", flush=True)

    wsis_dir = os.path.join(args.hest_dir, "wsis")
    for _, row in df.iterrows():
        extract_one_wsi(
            row["id"], wsis_dir, args.hest_dir, args.output_dir,
            crop_px=int(row["crop_px"]), out_px=args.out_px, seed=args.seed,
        )

    print("Extraction complete.", flush=True)


if __name__ == "__main__":
    main()
