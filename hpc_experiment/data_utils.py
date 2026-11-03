"""Shared dataset + model definitions for the HEST conditional diffusion experiment.

Imported by train.py, sample.py and compute_fid.py so the three scripts stay
in sync about the class ordering and model architecture.
"""

import glob
import os

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Fixed ordering so class index <-> cancer type mapping never drifts between
# training, sampling and FID computation.
CANCER_TYPES = [
    "CCRCC", "COAD", "HCC", "IDC", "LUNG",
    "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM",
]


def build_patch_index(hest_dir, cancer_types=CANCER_TYPES):
    """Build a global (file_path, local_idx, class_idx) index without loading
    any image data, so the pooled ~236K-patch dataset never needs to be fully
    read into RAM at once."""
    index = []
    for class_idx, ct in enumerate(cancer_types):
        files = sorted(glob.glob(os.path.join(hest_dir, ct, "patches", "*.h5")))
        if not files:
            print(f"[warn] no .h5 files found for {ct} under {hest_dir}")
        for fp in files:
            with h5py.File(fp, "r") as f:
                n = f["img"].shape[0]
            index.extend((fp, i, class_idx) for i in range(n))
    return index


class HestPatchDataset(Dataset):
    """Lazy-loading dataset over pooled HEST patches across cancer types.

    Opens .h5 files on demand (cached per worker process) instead of
    preloading everything into RAM. Necessary here because the pooled
    dataset is ~35GB raw -- fine to preload for a single cancer type (as in
    the exploratory notebooks) but not safe to assume for the full pool on
    an arbitrary compute node.
    """

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
        img = f["img"][local_idx]  # (224, 224, 3) uint8
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        t = F.interpolate(
            t.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        return t, class_idx

    def __getstate__(self):
        # h5py.File handles aren't safely picklable across DataLoader worker
        # processes -- drop them, each worker reopens files lazily on first access.
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class ClassConditionedUnet(nn.Module):
    """UNet2DModel wrapped with a class-embedding broadcast as extra input
    channels (same pattern as cdm.ipynb / cdm-hest.ipynb), here conditioned
    on cancer type instead of WSI identity."""

    def __init__(self, num_classes, image_size=64, class_emb_size=4):
        super().__init__()
        from diffusers import UNet2DModel  # local import keeps data_utils importable without diffusers for quick checks

        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3 + class_emb_size,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        c = self.class_emb(class_labels).view(bs, -1, 1, 1).expand(bs, -1, w, h)
        return self.model(torch.cat((x, c), dim=1), t).sample
