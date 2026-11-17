"""Shared dataset + model definitions for the image-to-cancer-type classifier
(pooled, 10 classes). Self-contained duplicate of the relevant pieces of
hpc_experiment/data_utils.py (CANCER_TYPES, patch indexing, dataset), plus a
small CNN classifier -- not imported across experiment folders, matching
this repo's existing convention.

Purpose: the class-conditioning analogue of hpc_experiment_regressor's
image->expression regressor. hpc_experiment/hpc_experiment_sr train on ALL
WSIs with no held-out split (unlike the RNA experiments, which use the
official HEST-bench WSI-level folds), so this module introduces its own
deterministic per-WSI train/test split (80/20, fixed seed) purely for
training and evaluating this classifier -- independent of, and not used by,
the generative models themselves.
"""

import glob
import os
import random

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

CANCER_TYPES = [
    "CCRCC", "COAD", "HCC", "IDC", "LUNG",
    "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM",
]


def _split_wsi_files(hest_dir, cancer_types, test_fraction=0.2, seed=0):
    """Deterministic per-cancer-type 80/20 split of WSI .h5 files, so the
    classifier's real-baseline accuracy is measured on WSIs it never
    trained on (the generative models have no such split -- this is purely
    for giving this classifier a fair, non-memorized baseline)."""
    rng = random.Random(seed)
    train_files, test_files = {}, {}
    for ct in cancer_types:
        files = sorted(glob.glob(os.path.join(hest_dir, ct, "patches", "*.h5")))
        files = files[:]
        rng.shuffle(files)
        n_test = max(1, int(len(files) * test_fraction))
        test_files[ct] = files[:n_test]
        train_files[ct] = files[n_test:]
    return train_files, test_files


def build_patch_index_split(hest_dir, cancer_types=CANCER_TYPES, test_fraction=0.2, seed=0):
    """Returns (train_index, test_index), each a list of
    (file_path, local_idx, class_idx) tuples, split by WSI (not by patch) to
    avoid leaking correlated patches from the same slide across the split."""
    train_files, test_files = _split_wsi_files(hest_dir, cancer_types, test_fraction, seed)

    def _index_for(files_by_class):
        index = []
        for class_idx, ct in enumerate(cancer_types):
            for fp in files_by_class[ct]:
                with h5py.File(fp, "r") as f:
                    n = f["img"].shape[0]
                index.extend((fp, i, class_idx) for i in range(n))
        return index

    return _index_for(train_files), _index_for(test_files)


class HestPatchDataset(Dataset):
    """Lazy-loading dataset over pooled HEST patches across cancer types."""

    def __init__(self, index, image_size=64):
        self.index = index
        self.image_size = image_size
        self._handles = {}

    def __len__(self):
        return len(self.index)

    def _get_handle(self, path):
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def __getitem__(self, idx):
        fp, local_idx, class_idx = self.index[idx]
        f = self._get_handle(fp)
        img = f["img"][local_idx]
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        t = F.interpolate(
            t.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        return t, class_idx

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class ImageToClassClassifier(nn.Module):
    """Small CNN: image -> cancer-type logits. Same backbone shape as
    hpc_experiment_regressor's ImageToExpressionRegressor (works at any
    input resolution via adaptive avg pool), softmax head instead of
    regression head."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
