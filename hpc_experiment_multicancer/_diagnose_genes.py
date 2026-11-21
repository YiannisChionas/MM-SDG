import anndata as ad
import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/selected_wsis.csv")

for code, sub in df.groupby("oncotree_code"):
    wsi_id = sub["id"].iloc[0]
    a = ad.read_h5ad(f"/opt/storage/datasets/HEST/st/{wsi_id}.h5ad")
    print(f"{code} (example {wsi_id}): n_vars={a.n_vars}, first 5 genes: {list(a.var_names[:5])}")
