#!/usr/bin/env python3
"""
Download helper script for OpenX datasets.

This script provides download instructions and utilities for downloading
OpenX datasets in RLDS format from TensorFlow Datasets.

Usage:
    python scripts/download_openx_datasets.py --base_dir <path> --dataset_name nyu_door_opening_surprising_effectiveness
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Dataset metadata
DATASET_INFO = {
    "bridge": {
        "full_name": "bridge_dataset",
        "tfds_name": "bridge_dataset",
        "num_trajectories": 25460,
        "description": "Bridge Dataset v2: A dataset for robot learning at scale",
        "size_estimate": "~100 GB",
    },
    "nyu_door_opening_surprising_effectiveness": {
        "full_name": "nyu_door_opening_surprising_effectiveness",
        "tfds_name": "nyu_door_opening_surprising_effectiveness",
        "num_trajectories": 435,
        "description": "NYU Door Opening: The surprising effectiveness of representation learning",
        "size_estimate": "~10 GB",
    },
    "berkeley_cable_routing": {
        "full_name": "berkeley_cable_routing",
        "tfds_name": "berkeley_cable_routing",
        "num_trajectories": 1482,
        "description": "UC Berkeley Cable Routing: Multi-stage cable routing through hierarchical imitation",
        "size_estimate": "~20 GB",
    },
    "berkeley_autolab_ur5": {
        "full_name": "berkeley_autolab_ur5",
        "tfds_name": "berkeley_autolab_ur5",
        "num_trajectories": 896,
        "description": "Berkeley AUTOLab UR5",
        "size_estimate": "~15 GB",
    },
}


def print_download_instructions(dataset_name: str, base_dir: Path):
    """Print manual download instructions for a dataset."""

    if dataset_name not in DATASET_INFO:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASET_INFO.keys())}")
        sys.exit(1)

    info = DATASET_INFO[dataset_name]
    dest_dir = base_dir / "rlds" / dataset_name

    print("=" * 80)
    print(f"DOWNLOAD INSTRUCTIONS: {dataset_name}")
    print("=" * 80)
    print(f"\nDataset: {info['description']}")
    print(f"Trajectories: {info['num_trajectories']}")
    print(f"Estimated Size: {info['size_estimate']}")
    print(f"\nDestination: {dest_dir}")
    print("\n" + "=" * 80)
    print("METHOD 1: Using tensorflow_datasets (Recommended)")
    print("=" * 80)
    print("\nRun the following Python code:\n")
    print("```python")
    print("import tensorflow_datasets as tfds")
    print(f"import os")
    print()
    print("# Set download directory")
    print(f"os.environ['TFDS_DATA_DIR'] = '{base_dir / 'rlds'}'")
    print()
    print("# Download the dataset")
    print(f"builder = tfds.builder('{info['tfds_name']}')")
    print("builder.download_and_prepare()")
    print("```")

    print("\n" + "=" * 80)
    print("METHOD 2: Using tfds command line")
    print("=" * 80)
    print(f"\nRun this command:\n")
    print(f"tfds build {info['tfds_name']} --data_dir={base_dir / 'rlds'}")

    print("\n" + "=" * 80)
    print("METHOD 3: Manual download from Google Cloud")
    print("=" * 80)
    print("\n1. Install gsutil: https://cloud.google.com/storage/docs/gsutil_install")
    print(f"2. Download dataset:\n")
    print(f"   gsutil -m cp -r gs://gresearch/robotics/{info['tfds_name']}/* {dest_dir}/")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"\nAfter download, verify the dataset exists at:")
    print(f"  {dest_dir}")
    print("\nThe directory should contain:")
    print("  - features.json")
    print("  - dataset_info.json")
    print("  - One or more .tfrecord files")
    print("\nRun verification with:")
    print(f"  python scripts/download_openx_datasets.py --base_dir {base_dir} --dataset_name {dataset_name} --verify")
    print("\n" + "=" * 80)


def verify_dataset(dataset_name: str, base_dir: Path) -> bool:
    """Verify that a dataset has been downloaded correctly."""

    if dataset_name not in DATASET_INFO:
        print(f"Error: Unknown dataset '{dataset_name}'")
        return False

    dest_dir = base_dir / "rlds" / dataset_name

    print(f"\nVerifying dataset: {dataset_name}")
    print(f"Location: {dest_dir}")

    if not dest_dir.exists():
        print(f"❌ Dataset directory does not exist: {dest_dir}")
        return False

    print(f"✓ Dataset directory exists")

    # Check for required files
    required_patterns = ["*.tfrecord*", "dataset_info.json"]
    found_files = []

    for pattern in required_patterns:
        files = list(dest_dir.glob(f"**/{pattern}"))
        if files:
            print(f"✓ Found {len(files)} file(s) matching '{pattern}'")
            found_files.extend(files)
        else:
            print(f"⚠ Warning: No files matching '{pattern}' found")

    if not found_files:
        print(f"❌ No dataset files found in {dest_dir}")
        return False

    # Try to load with tensorflow_datasets
    try:
        import tensorflow_datasets as tfds
        print("\nAttempting to load dataset with tfds...")
        builder = tfds.builder_from_directory(str(dest_dir))
        info = builder.info
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Splits: {list(info.splits.keys())}")
        print(f"  - Features: {list(info.features.keys())}")

        # Check if we can iterate
        ds = builder.as_dataset(split='train[:1]')
        sample = next(iter(ds))
        print(f"✓ Successfully loaded sample trajectory")
        print(f"  - Keys: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"⚠ Warning: Could not load dataset with tfds: {e}")
        print(f"   Dataset files exist but may need validation")
        return False


def create_directory_structure(base_dir: Path):
    """Create the required directory structure for datasets."""

    directories = [
        base_dir / "rlds",
        base_dir / "vla",
        base_dir / "hdf5",
        base_dir / "hf",
        base_dir / "cache",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def list_datasets():
    """List all available datasets."""

    print("\n" + "=" * 80)
    print("AVAILABLE OPENX DATASETS")
    print("=" * 80)

    for name, info in DATASET_INFO.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Trajectories: {info['num_trajectories']}")
        print(f"  Estimated Size: {info['size_estimate']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download helper for OpenX datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for dataset storage (e.g., /data/fog_x or ~/robodm/data)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=list(DATASET_INFO.keys()),
        help="Name of the dataset to download",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that the dataset has been downloaded correctly",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create directory structure only",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()

    if args.list:
        list_datasets()
        return

    if args.setup:
        print(f"\nSetting up directory structure at: {base_dir}")
        create_directory_structure(base_dir)
        print(f"\n✓ Directory structure created successfully!")
        return

    if not args.dataset_name:
        parser.print_help()
        print("\n")
        list_datasets()
        return

    if args.verify:
        success = verify_dataset(args.dataset_name, base_dir)
        sys.exit(0 if success else 1)
    else:
        # Create directory structure first
        create_directory_structure(base_dir)
        # Print download instructions
        print_download_instructions(args.dataset_name, base_dir)


if __name__ == "__main__":
    main()
