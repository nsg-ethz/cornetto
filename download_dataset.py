"""
Download the Cornetto benchmark dataset from HuggingFace.

Run once after cloning:
    python download_dataset.py

The dataset is saved to dataset/main_dataset/.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ID = "iprotogeros/cornetto-benchmark"
HF_BASE = "https://huggingface.co/datasets"
DEFAULT_DEST = Path(__file__).parent / "dataset" / "main_dataset"


def main():
    parser = argparse.ArgumentParser(description="Download the Cornetto dataset")
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (only needed if the dataset is private). "
             "Alternatively set the HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_DEST),
        help=f"Where to save the dataset (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()

    dest = Path(args.local_dir)
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} already exists and is non-empty — skipping download.")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        clone_url = f"https://user:{token}@huggingface.co/datasets/{REPO_ID}"
    else:
        clone_url = f"{HF_BASE}/{REPO_ID}"

    print(f"Cloning {REPO_ID} (this may take a few minutes) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["git", "clone", "--depth=1", clone_url, tmpdir],
            check=True,
        )
        src = Path(tmpdir) / "main_dataset"
        if not src.exists():
            raise RuntimeError(
                "Expected 'main_dataset/' in the cloned repo but it was not found."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))

    print(f"Done. Dataset saved to {dest}")


if __name__ == "__main__":
    main()
