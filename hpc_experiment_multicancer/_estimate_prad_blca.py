import pandas as pd

MAX_TILES = 4000

df = pd.read_csv("/opt/storage/datasets/HEST/HEST_v1_3_0.csv")
visium_human = df[(df["st_technology"] == "Visium") & (df["species"] == "Homo sapiens")]

for code in ["PRAD", "BLCA"]:
    sub = visium_human[visium_human["oncotree_code"] == code].copy()
    sub["capped"] = sub["spots_under_tissue"].clip(upper=MAX_TILES)
    print(f"\n=== {code} ({len(sub)} human Visium WSIs) ===")
    print(sub[["id", "spots_under_tissue", "pixel_size_um_estimated"]].sort_values("spots_under_tissue", ascending=False).to_string(index=False))
    print(f"Total (uncapped): {sub['spots_under_tissue'].sum():.0f}")
    print(f"Total (capped at {MAX_TILES}/WSI): {sub['capped'].sum():.0f}")
    print(f"Mean/WSI (capped): {sub['capped'].mean():.0f}")
