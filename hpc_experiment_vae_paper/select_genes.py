"""Step 0: select the top-K highly-variable genes (by default K=17,655, to
match the RNA-CDM paper's VAE input size) from the CCRCC train-split WSIs'
full ~36,601-gene transcriptome -- HEST-bench's own var_50genes.json panel
is a small curated benchmark subset, not a stand-in for "most variable
genes at paper scale".

CPU-only (no model, just mean/variance accumulation over all spots) --
run as its own lightweight job, no GPU needed. Saves the panel so
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
    parser.add_argument("--top-k", type=int, default=17655)
    parser.add_argument("--min-detection-frac", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_sample_ids(args.data_dir, CANCER_TYPE, args.fold, "train")
    print(f"Fold {args.fold} train WSIs ({len(train_ids)}): {train_ids}", flush=True)

    genes = select_highly_variable_genes(
        args.data_dir, train_ids, CANCER_TYPE,
        top_k=args.top_k, min_detection_frac=args.min_detection_frac,
    )

    panel_path = out_dir / f"gene_panel_{len(genes)}.json"
    save_gene_panel(panel_path, genes)
    print(f"Saved {len(genes)}-gene panel to {panel_path}", flush=True)


if __name__ == "__main__":
    main()
