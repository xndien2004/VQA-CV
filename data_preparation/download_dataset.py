"""
Download VQA datasets from HuggingFace using snapshot_download,
then unzip images.zip in each dataset folder.

Datasets:
  - nhonhoccode/RecieptVQA
  - nhonhoccode/ViOCRVQA
  - nhonhoccode/ViTextVQA

Usage:
  python scripts/download_dataset.py [--output_dir ./datasets] [--dataset all|recieptvqa|viocrvqa|vitextvqa]
"""

import argparse
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download


DATASETS = {
    "recieptvqa": "nhonhoccode/RecieptVQA",
    "viocrvqa":   "nhonhoccode/ViOCRVQA",
    "vitextvqa":  "nhonhoccode/ViTextVQA",
}


def unzip_images(dataset_dir: Path):
    """Unzip all images.zip files found in the dataset directory."""
    zip_files = list(dataset_dir.rglob("images.zip"))
    if not zip_files:
        print(f"  No images.zip found in {dataset_dir}")
        return

    for zip_path in zip_files:
        extract_to = zip_path.parent
        print(f"  Unzipping {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        print(f"  Done unzip: {zip_path.name}")


def download_dataset(name: str, hf_repo: str, output_dir: Path):
    local_dir = output_dir / name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] Downloading {hf_repo} -> {local_dir}")
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"[{name}] Download complete.")

    unzip_images(local_dir)
    print(f"[{name}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Download VQA datasets from HuggingFace")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Root directory to save datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + list(DATASETS.keys()),
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    targets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    for name, hf_repo in targets.items():
        download_dataset(name, hf_repo, output_dir)

    print("\nAll downloads complete.")
    print(f"Dataset root: {output_dir}")


if __name__ == "__main__":
    main()
