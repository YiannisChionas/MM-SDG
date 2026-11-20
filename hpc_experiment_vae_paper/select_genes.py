"""Step 0: select genes from the CCRCC train-split WSIs' full ~36,601-gene
transcriptome, using scanpy's standard QC filter (sc.pp.filter_genes,
min_cells as a fraction of total spots) -- HEST-bench's own
var_50genes.json panel is a small curated benchmark subset, not usable at
paper scale.

Deliberately NO highly-variable-gene cutoff on top of the QC filter: we
measured that an HVG dispersion threshold (flavor='seurat', default
thresholds) kept only 2,913/14,166 QC-passed genes while capturing just
22% of total expression variance -- too lossy for a generative model's
conditioning signal. The QC filter alone keeps 99.9% of total raw counts,
so this is close to "all usable genes", letting the VAE's own bottleneck
(latent_dim) do the real compression, matching the paper's own approach
of feeding its VAE a large, lightly filtered gene set (their 17,655 was
itself just whatever their own preprocessing left them with, not a
deliberately small curated panel).

CPU-only (no model, just scanpy QC filtering over pooled spots) -- run as
its own lightweight job, no GPU needed. Saves the panel so
train_vae_paper.py can be re-run deterministically against the exact
same gene list.
"""

import argparse
from pathlib import Path

from data_utils_vae_paper import (
    CANCER_TYPE, load_split_sample_ids, select_qc_passed_genes, save_gene_panel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min-cells-frac", type=float, default=0.01,
                         help="scanpy sc.pp.filter_genes min_cells, as a fraction of total spots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    print(f"Fold {args.fold} train WSIs ({len(train_ids)}): {train_ids}", flush=True)

    genes = select_qc_passed_genes(args.data_dir, train_ids, CANCER_TYPE, min_cells_frac=args.min_cells_frac)

    panel_path = out_dir / f"gene_panel_{len(genes)}.json"
    save_gene_panel(panel_path, genes)
    print(f"Saved {len(genes)}-gene panel to {panel_path}", flush=True)


if __name__ == "__main__":
    main()
