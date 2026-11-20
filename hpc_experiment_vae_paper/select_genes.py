"""Step 0: select highly-variable genes from the CCRCC train-split WSIs'
full ~36,601-gene transcriptome, using scanpy's standard HVG procedure
(flavor='seurat', the field's common default) -- HEST-bench's own
var_50genes.json panel is a small curated benchmark subset, not a
stand-in for "highly variable genes" at paper scale.

The resulting panel size is NOT forced to the RNA-CDM paper's 17,655 --
that number was itself just whatever the paper's own preprocessing left
them with (their own wording), not a target they chose. We apply a
standard, defensible selection method instead and report whatever gene
count results; if it ends up being more or fewer than 17,655, that's
fine -- see this experiment's README.

CPU-only (no model, just scanpy preprocessing over pooled spots) -- run
as its own lightweight job, no GPU needed. Saves the panel so
train_vae_paper.py can be re-run deterministically against the exact same
gene list.
"""

import argparse
from pathlib import Path

from data_utils_vae_paper import (
    CANCER_TYPE, load_split_sample_ids, select_highly_variable_genes, save_gene_panel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min-cells-frac", type=float, default=0.01,
                         help="scanpy sc.pp.filter_genes min_cells, as a fraction of total spots")
    parser.add_argument("--min-mean", type=float, default=0.0125, help="scanpy highly_variable_genes min_mean")
    parser.add_argument("--max-mean", type=float, default=3.0, help="scanpy highly_variable_genes max_mean")
    parser.add_argument("--min-disp", type=float, default=0.5, help="scanpy highly_variable_genes min_disp")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    print(f"Fold {args.fold} train WSIs ({len(train_ids)}): {train_ids}", flush=True)

    genes = select_highly_variable_genes(
        args.data_dir, train_ids, CANCER_TYPE,
        min_cells_frac=args.min_cells_frac, min_mean=args.min_mean,
        max_mean=args.max_mean, min_disp=args.min_disp,
    )

    panel_path = out_dir / f"gene_panel_{len(genes)}.json"
    save_gene_panel(panel_path, genes)
    print(f"Saved {len(genes)}-gene panel to {panel_path}", flush=True)


if __name__ == "__main__":
    main()
