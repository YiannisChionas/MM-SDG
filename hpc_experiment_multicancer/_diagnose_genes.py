import anndata as ad
import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/selected_wsis.csv")

print("=== n_vars for ALL 47 selected WSIs ===")
rows = []
for _, row in df.iterrows():
    wsi_id, code = row["id"], row["oncotree_code"]
    a = ad.read_h5ad(f"/opt/storage/datasets/HEST/st/{wsi_id}.h5ad")
    rows.append({"id": wsi_id, "oncotree_code": code, "n_vars": a.n_vars})

out = pd.DataFrame(rows).sort_values("n_vars")
print(out.to_string(index=False))
print(f"\nmin n_vars: {out['n_vars'].min()} ({out.iloc[0]['id']}, {out.iloc[0]['oncotree_code']})")
