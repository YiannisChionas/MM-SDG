"""Local smoke test -- NOT part of the HPC experiment, just a quick sanity
check that data_utils/train/sample work end-to-end before burning cluster
GPU-hours on a real bug. Deleted after use.

num_workers=0 here: the real cluster runs Linux (fork), but this dev box is
Windows (spawn) -- torch DataLoader workers under spawn re-import the
launching script, so without a __main__ guard it recurses forever. train.py
itself already has that guard for real runs; this harness avoids the whole
issue by just testing single-process data loading + a separate pickle check
for the Linux-fork code path (HestPatchDataset.__getstate__).
"""

import pickle
import shutil
import sys
import tempfile
from pathlib import Path


def main():
    ROOT = Path(__file__).parent
    REAL_HEST = ROOT.parent / "hest-bench"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "hest-bench"

        for ct in ["HCC", "SKCM"]:
            src_dir = REAL_HEST / ct / "patches"
            files = sorted(src_dir.glob("*.h5"))
            assert files, f"no files found for {ct}"
            dst_dir = data_dir / ct / "patches"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(files[0], dst_dir / files[0].name)
            print(f"staged {ct}: {files[0].name}")

        out_dir = tmp / "exp"

        sys.path.insert(0, str(ROOT))
        import data_utils
        data_utils.CANCER_TYPES = ["HCC", "SKCM"]

        print("\n=== Check: HestPatchDataset is picklable (Linux-fork worker safety) ===")
        index = data_utils.build_patch_index(str(data_dir), data_utils.CANCER_TYPES)
        ds = data_utils.HestPatchDataset(index, image_size=64)
        _ = ds[0]  # force-open a handle so pickling actually has something to drop
        pickled = pickle.dumps(ds)
        ds2 = pickle.loads(pickled)
        assert ds2._handles == {}, "unpickled dataset should start with no open handles"
        x, y = ds2[0]
        assert x.shape == (3, 64, 64)
        print("pickle/unpickle OK, sample shape", tuple(x.shape), "label", y)

        import train as train_mod

        print("\n=== Smoke test: train.py (1 epoch, tiny subset, num_workers=0) ===")
        sys.argv = [
            "train.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(out_dir),
            "--epochs", "1",
            "--batch-size", "8",
            "--num-workers", "0",
        ]
        train_mod.main()

        assert (out_dir / "checkpoint.pt").exists(), "checkpoint.pt was not created"
        print("checkpoint.pt created OK")

        print("\n=== Smoke test: resume (call again with epochs=2) ===")
        sys.argv = [
            "train.py",
            "--data-dir", str(data_dir),
            "--output-dir", str(out_dir),
            "--epochs", "2",
            "--batch-size", "8",
            "--num-workers", "0",
        ]
        train_mod.main()
        print("resume OK")

        print("\n=== Smoke test: sample.py (2 images/class) ===")
        import sample as sample_mod
        sys.argv = [
            "sample.py",
            "--checkpoint", str(out_dir / "checkpoint.pt"),
            "--output-dir", str(tmp / "synthetic"),
            "--per-class", "2",
            "--batch-size", "2",
        ]
        sample_mod.main()

        synth_files = list((tmp / "synthetic").rglob("*.png"))
        assert len(synth_files) == 4, f"expected 4 synthetic images, got {len(synth_files)}"
        print(f"sample.py OK -- generated {len(synth_files)} images")

        print("\n=== Smoke test: sample.py resume (per-class=3, should only add 1 each) ===")
        sys.argv = [
            "sample.py",
            "--checkpoint", str(out_dir / "checkpoint.pt"),
            "--output-dir", str(tmp / "synthetic"),
            "--per-class", "3",
            "--batch-size", "2",
        ]
        sample_mod.main()
        synth_files = list((tmp / "synthetic").rglob("*.png"))
        assert len(synth_files) == 6, f"expected 6 synthetic images after resume, got {len(synth_files)}"
        print(f"sample.py resume OK -- now {len(synth_files)} images")

        print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
