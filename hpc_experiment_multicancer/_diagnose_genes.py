import anndata as ad
import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/selected_wsis.csv")

print("=== var columns + gene_ids check per type ===")
gene_id_sets = {}
symbol_sets = {}
for code, sub in df.groupby("oncotree_code"):
    wsi_id = sub["id"].iloc[0]
    a = ad.read_h5ad(f"/opt/storage/datasets/HEST/st/{wsi_id}.h5ad")
    print(f"{code} (example {wsi_id}): var columns = {list(a.var.columns)}")
    symbol_sets[code] = set(a.var_names)
    if "gene_ids" in a.var.columns:
        gene_id_sets[code] = set(a.var["gene_ids"])
        print(f"  first 3 gene_ids: {list(a.var['gene_ids'][:3])}")

print("\n=== Intersection by gene SYMBOL (var_names) ===")
common_symbols = set.intersection(*symbol_sets.values())
print(f"Common across all 5 types: {len(common_symbols)}")

if len(gene_id_sets) == len(symbol_sets):
    print("\n=== Intersection by Ensembl gene_ids ===")
    common_ids = set.intersection(*gene_id_sets.values())
    print(f"Common across all 5 types: {len(common_ids)}")
else:
    print(f"\ngene_ids not available for all types ({len(gene_id_sets)}/{len(symbol_sets)}), skipping Ensembl check")
