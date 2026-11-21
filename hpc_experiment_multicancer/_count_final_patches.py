import h5py
import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/selected_wsis.csv")
wsi_to_code = dict(zip(df["id"], df["oncotree_code"]))

totals = {}
for wsi_id, code in wsi_to_code.items():
    path = f"/opt/storage/datasets/HEST/patches_multicancer/{wsi_id}.h5"
    with h5py.File(path, "r") as f:
        n = f["img"].shape[0]
    totals[code] = totals.get(code, 0) + n
    print(f"{wsi_id} ({code}): {n} patches")

print("\n--- Per-type totals (final, post-extraction, post-4K-cap) ---")
grand_total = 0
for code, n in sorted(totals.items()):
    print(f"{code}: {n}")
    grand_total += n
print(f"TOTAL: {grand_total}")
