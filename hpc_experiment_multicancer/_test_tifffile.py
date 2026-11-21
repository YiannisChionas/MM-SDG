"""Throwaway -- confirm tifffile can do efficient region-crop reads on a
real HEST WSI tiff without loading the whole image into memory."""
import time

import numpy as np
import tifffile

PATH = "/opt/storage/datasets/HEST/wsis/INT24.tif"

t0 = time.time()
tif = tifffile.TiffFile(PATH)
print(f"Opened in {time.time()-t0:.3f}s")
page = tif.pages[0]
print("Shape:", page.shape, "dtype:", page.dtype)
print("Is tiled:", page.is_tiled)
print("Tile size:", getattr(page, "tilewidth", None), getattr(page, "tilelength", None))

# Memory-mapped array -- should NOT load the full image into RAM
t0 = time.time()
arr = tifffile.memmap(PATH)
print(f"memmap() in {time.time()-t0:.3f}s, shape={arr.shape}")

# Crop a 280x280 region around some pixel coordinate (simulating a spot crop)
t0 = time.time()
row, col = 10000, 10000
half = 140
crop = np.array(arr[row - half:row + half, col - half:col + half])
print(f"Cropped {crop.shape} region in {time.time()-t0:.3f}s")
print("Crop stats: min", crop.min(), "max", crop.max(), "mean", crop.mean())
