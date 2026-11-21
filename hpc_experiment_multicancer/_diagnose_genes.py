import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/HEST_v1_3_0.csv")
visium = df[df["st_technology"] == "Visium"]
target = visium[visium["oncotree_code"].isin(["COAD", "SCCRCC", "EPM", "LIHB", "IDC"])]

print("species x oncotree_code (Visium, our 5 target types):")
print(pd.crosstab(target["oncotree_code"], target["species"]))

print("\nAll species values across the full Visium pool:")
print(visium["species"].value_counts())
