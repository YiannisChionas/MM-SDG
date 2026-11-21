import pandas as pd

df = pd.read_csv("/opt/storage/datasets/HEST/selected_wsis.csv")
for code, sub in df.groupby("oncotree_code"):
    has_patient = sub["patient"].notna().all()
    if has_patient:
        n_patients = sub["patient"].nunique()
        print(f"{code}: {len(sub)} WSIs, {n_patients} unique patients, "
              f"avg {len(sub)/n_patients:.2f} WSIs/patient")
    else:
        print(f"{code}: {len(sub)} WSIs, patient field EMPTY -- no patient info")
