import numpy as np
import pandas as pd
import torch

from data_utils_multicancer import (
    split_train_test_wsis, build_expr_index, GeneExpressionNormalizer,
    HestMultiCancerExpressionDataset, BetaVAEPaper, load_gene_panel,
)

HEST_DIR = "/opt/storage/datasets/HEST"
CKPT_PATH = "/opt/storage/users/it219111/experiments/hest-vae-multicancer/vae_multicancer_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CKPT_PATH, map_location=device)
genes = ckpt["genes"]
normalizer = GeneExpressionNormalizer.from_state_dict(ckpt["normalizer"])

model = BetaVAEPaper(
    input_dim=ckpt["input_dim"], hidden1=ckpt["hidden1"], hidden2=ckpt["hidden2"], latent_dim=ckpt["latent_dim"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

wsis_df = pd.read_csv(f"{HEST_DIR}/selected_wsis.csv")
_, test_ids = split_train_test_wsis(wsis_df, test_frac=0.2, seed=0)
print(f"Test WSIs: {len(test_ids)}", flush=True)

wsi_to_code = dict(zip(wsis_df["id"], wsis_df["oncotree_code"]))
test_index = build_expr_index(HEST_DIR, test_ids, wsi_to_code, genes)
print(f"Test spots: {len(test_index)}", flush=True)

ds = HestMultiCancerExpressionDataset(test_index, normalizer)

# Sparsity/skew sanity check on the z-scored test data itself.
all_z = np.stack([normalizer.transform(e) for e, _ in test_index], axis=0)
print(f"\nz-scored data stats: mean={all_z.mean():.4f} std={all_z.std():.4f} "
      f"min={all_z.min():.2f} max={all_z.max():.2f}", flush=True)
print(f"Fraction of |z| > 5: {(np.abs(all_z) > 5).mean()*100:.2f}%", flush=True)
print(f"Fraction of |z| > 10: {(np.abs(all_z) > 10).mean()*100:.2f}%", flush=True)
per_gene_std_of_raw = all_z.std(axis=0)  # should be ~1 per gene by construction on FIT set, check drift on test set
print(f"Per-gene z std on TEST set: mean={per_gene_std_of_raw.mean():.3f}, "
      f"min={per_gene_std_of_raw.min():.3f}, max={per_gene_std_of_raw.max():.3f}", flush=True)

loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
sq_errors, abs_errors, n = 0.0, 0.0, 0
with torch.no_grad():
    for x, _codes in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        sq_errors += ((recon - x) ** 2).sum().item()
        abs_errors += (recon - x).abs().sum().item()
        n += x.numel()

rmse = (sq_errors / n) ** 0.5
mae = abs_errors / n
print(f"\nTEST SET (best checkpoint, epoch {ckpt['epoch']}): RMSE={rmse:.4f}, MAE={mae:.4f}", flush=True)
print(f"(paper reports RMSE=0.2475, MAE=0.1471 on their test set)", flush=True)
