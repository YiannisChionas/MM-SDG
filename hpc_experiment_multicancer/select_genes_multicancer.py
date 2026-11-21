"""Gene panel selection for the multi-cancer pool (DECISIONS.md #8):
genes common across ALL 50 selected WSIs (5 cancer types), then a
standard scanpy QC filter only -- NO highly-variable-gene cutoff, same
reasoning as hpc_experiment_vae_paper (measured there that an HVG
dispersion threshold discards too much real expression variance for a
generative model's conditioning input; see that experiment's README).

Matches the paper's own stated method: "we selected those genes the
different cancer types have in common" (intersection), then basic
filtering -- the resulting panel SIZE is whatever this yields, not a
chosen target.

Pools over ALL 50 selected WSIs (not yet restricted to a train split --
train/val/test split for this multi-cancer pool hasn't been decided yet,
see DECISIONS.md's open items; this gene panel may need recomputing on
the train-only subset once that split exists, to avoid any leakage of
val/test statistics into the panel).
"""

import argparse
import json
import os

import anndata as ad
import pandas as pd


def select_common_qc_genes(adata_dir, wsi_ids, min_cells_frac=0.01):
    import scanpy as sc

    adatas = []
    for wsi_id in wsi_ids:
        adata_path = os.path.join(adata_dir, "st", f"{wsi_id}.h5ad")
        if not os.path.exists(adata_path):
            print(f"[warn] missing adata for {wsi_id}, skipping")
            continue
        a = ad.read_h5ad(adata_path)
        a.obs_names = [f"{wsi_id}_{bc}" for bc in a.obs_names]
        adatas.append(a)

    pooled = ad.concat(adatas, join="inner")  # inner = intersection of gene sets across all 50 WSIs
    print(f"Pooled {pooled.n_obs} spots x {pooled.n_vars} common genes across {len(adatas)} WSIs", flush=True)

    min_cells = max(1, int(min_cells_frac * pooled.n_obs))
    sc.pp.filter_genes(pooled, min_cells=min_cells)
    print(f"After filter_genes(min_cells={min_cells}): {pooled.n_vars} genes kept", flush=True)

    return pooled.var_names.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-wsis-csv", type=str, required=True)
    parser.add_argument("--adata-dir", type=str, required=True, help="Dir containing st/*.h5ad")
    parser.add_argument("--min-cells-frac", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.selected_wsis_csv)
    wsi_ids = df["id"].tolist()
    print(f"Selecting common genes across {len(wsi_ids)} WSIs (5 cancer types)", flush=True)

    genes = select_common_qc_genes(args.adata_dir, wsi_ids, args.min_cells_frac)

    os.makedirs(args.output_dir, exist_ok=True)
    panel_path = os.path.join(args.output_dir, f"gene_panel_multicancer_{len(genes)}.json")
    with open(panel_path, "w", encoding="utf-8") as f:
        json.dump({"genes": genes}, f)
    print(f"Saved {len(genes)}-gene panel to {panel_path}", flush=True)


if __name__ == "__main__":
    main()
