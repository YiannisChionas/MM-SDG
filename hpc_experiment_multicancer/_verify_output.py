import h5py
import anndata as ad

with h5py.File("/opt/storage/datasets/HEST/patches_multicancer/INT16.h5", "r") as f:
    print("Keys:", list(f.keys()))
    print("img shape:", f["img"].shape, f["img"].dtype)
    print("barcode shape:", f["barcode"].shape, f["barcode"].dtype)
    print("First 3 barcodes:", f["barcode"][:3])
    img0 = f["img"][0]
    print("img[0] stats: min", img0.min(), "max", img0.max(), "mean", img0.mean())
    bc0_raw = f["barcode"][0][0]
    bc0 = bc0_raw.decode() if isinstance(bc0_raw, bytes) else str(bc0_raw)

a = ad.read_h5ad("/opt/storage/datasets/HEST/st/INT16.h5ad")
print("Barcode present in adata obs_names:", bc0 in a.obs_names)
