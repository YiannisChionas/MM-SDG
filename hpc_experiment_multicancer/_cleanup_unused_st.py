import os

import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/selected_wsis.csv")
selected_ids = set(df["id"])

st_dir = "/opt/storage/datasets/HEST/st"
removed, total_bytes = 0, 0
for fname in os.listdir(st_dir):
    if not fname.endswith(".h5ad"):
        continue
    wsi_id = fname[: -len(".h5ad")]
    if wsi_id in selected_ids:
        continue
    path = os.path.join(st_dir, fname)
    total_bytes += os.path.getsize(path)
    os.remove(path)
    removed += 1
    print(f"removed {fname}")

print(f"\nRemoved {removed} unused .h5ad files, freed {total_bytes / 1e9:.2f} GB")
