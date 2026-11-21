"""Throwaway -- confirm tifffile+zarr can do efficient region-crop reads
on a real HEST WSI tiff without loading the whole image into memory."""
import time

import numpy as np
import tifffile
import zarr

PATH = "/opt/storage/datasets/HEST/wsis/INT24.tif"

t0 = time.time()
store = tifffile.imread(PATH, aszarr=True, key=0)
z = zarr.open(store, mode="r")
print(f"Opened as zarr in {time.time()-t0:.3f}s, shape={z.shape}, dtype={z.dtype}, chunks={z.chunks}")

t0 = time.time()
row, col = 10000, 10000
half = 140
crop = np.asarray(z[row - half:row + half, col - half:col + half])
print(f"Cropped {crop.shape} region in {time.time()-t0:.3f}s")
print("Crop stats: min", crop.min(), "max", crop.max(), "mean", crop.mean())

# A second crop far away, to confirm it's doing lazy tile reads each time
# (not caching the whole image), by checking timing stays similarly fast.
t0 = time.time()
crop2 = np.asarray(z[500:500 + 280, 500:500 + 280])
print(f"Second cropped {crop2.shape} region in {time.time()-t0:.3f}s")

store.close()
print("OK")
