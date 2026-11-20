"""Shared dataset + model definitions for the cascaded super-resolution
(stage 2) experiment: conditions on BOTH the cancer-type class label AND a
low-resolution image, to produce the native 224x224 patch resolution --
matching hpc_experiment's real patches directly, no downsampling needed.

Fixes a structural issue identified while reviewing hpc_experiment's FID:
that experiment trains/generates at 64x64, then pytorch-fid has to upsample
64->299 for InceptionV3 (a 4.67x blow-up) while the paper's cascaded model
works natively at 256x256 (only 1.17x to Inception's 299). This stage
produces 224x224 directly (HEST patches' native size), so FID evaluation
here only needs a 224->299 (1.33x) resize -- much closer to the paper's
protocol.

Mirrors hpc_experiment/data_utils.py's structure (same CANCER_TYPES,
build_patch_index) but adds the paired low/high-res dataset and the
dual-conditioned UNet.
"""

import glob
import os

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

CANCER_TYPES = [
    "CCRCC", "COAD", "HCC", "IDC", "LUNG",
    "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM",
]


def build_patch_index(hest_dir, cancer_types=CANCER_TYPES):
    """Identical to hpc_experiment/data_utils.py's version -- same pooled,
    lazy (file_path, local_idx, class_idx) index, no image data loaded yet."""
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


class HestPatchPairDataset(Dataset):
    """Returns (low_res_condition, high_res_target, class_idx).

    high_res_target: the real patch at native resolution (224x224 for HEST
    -- no resize needed, unlike hpc_experiment's forced downsample to 64).

    low_res_condition: the SAME real patch, degraded by downsampling to
    low_res_size then upsampling back to image_size. This simulates what
    stage-1's (imperfect, 64x64) generated output looks like once upsampled
    for stage-2 conditioning -- standard practice in cascaded diffusion
    training (Ho et al., "Cascaded Diffusion Models"): train stage-2 on
    artificially-degraded REAL images, since stage-1's actual outputs are
    only needed at inference/sampling time, not training time.

    Conditioning augmentation: on top of the clean down->up degradation,
    add random-strength Gaussian noise to low_res_condition (std sampled
    per-example, uniform in [0, cond_aug_max_std]). Without this, stage-2
    only ever sees a PERFECT degradation of a real image during training,
    but at sampling time it's conditioned on stage-1's actual DDPM output,
    which carries its own imperfections/artifacts -- a train/inference
    distribution mismatch that Ho et al. and Imagen both address exactly
    this way. cond_aug_max_std=0 disables it (recovers the old behaviour).
    """

    def __init__(self, index, image_size=224, low_res_size=64, cond_aug_max_std=0.2):
        self.index = index
        self.image_size = image_size
        self.low_res_size = low_res_size
        self.cond_aug_max_std = cond_aug_max_std
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
        img = f["img"][local_idx]  # (224, 224, 3) uint8 -- HEST's native patch size
        high_res = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        if high_res.shape[-1] != self.image_size:
            high_res = F.interpolate(
                high_res.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        low_res = F.interpolate(
            high_res.unsqueeze(0), size=(self.low_res_size, self.low_res_size),
            mode="bilinear", align_corners=False,
        )
        low_res_upsampled = F.interpolate(
            low_res, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        if self.cond_aug_max_std > 0:
            noise_std = torch.rand(1).item() * self.cond_aug_max_std
            low_res_upsampled = low_res_upsampled + torch.randn_like(low_res_upsampled) * noise_std

        return low_res_upsampled, high_res, class_idx

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class SRConditionedUnet(nn.Module):
    """UNet2DModel conditioned on BOTH a low-res image (concatenated as 3
    extra input channels) and a cancer-type class embedding (broadcast as
    4 more channels, same mechanism as hpc_experiment's ClassConditionedUnet).

    Deliberately smaller channel width (64 base) than hpc_experiment's
    64x64 model (128 base) despite the higher 224x224 resolution -- mirrors
    the paper's own strategy (their stage-2 UNet dim=128 vs stage-1's 256).
    Confirmed via local benchmark this is the only config that fits a
    realistic cluster time budget at this resolution (28.5M params).
    """

    def __init__(self, num_classes, image_size=224, class_emb_size=4):
        super().__init__()
        from diffusers import UNet2DModel

        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=3 + 3 + class_emb_size,  # noisy high-res + low-res cond + class
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
        )

    def forward(self, noisy_high_res, low_res_cond, t, class_labels):
        bs, ch, w, h = noisy_high_res.shape
        c = self.class_emb(class_labels).view(bs, -1, 1, 1).expand(bs, -1, w, h)
        net_input = torch.cat((noisy_high_res, low_res_cond, c), dim=1)
        return self.model(net_input, t).sample


class Stage1ClassConditionedUnet(nn.Module):
    """Exact copy of hpc_experiment/data_utils.py's ClassConditionedUnet
    (64x64, 114M params) -- duplicated here (not imported across folders)
    so this experiment stays self-contained, and so sample_sr.py can load
    hpc_experiment's already-trained checkpoint.pt directly to produce the
    low-res image for cascaded sampling, matching the paper's two-stage
    inference: stage 1 generates 64x64, stage 2 refines to 224x224."""

    def __init__(self, num_classes, image_size=64, class_emb_size=4):
        super().__init__()
        from diffusers import UNet2DModel

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
