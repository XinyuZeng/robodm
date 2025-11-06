#!/usr/bin/env python3
"""
Download pre-converted LeRobot datasets from HuggingFace Hub.

This downloads already-converted datasets instead of converting from RLDS.

Usage:
    python scripts/download_hf_dataset.py --base_dir <path> --dataset_name nyu_door_opening_surprising_effectiveness
"""

import argparse
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mapping of dataset names to HuggingFace repos
HF_DATASET_REPOS = {
    "nyu_door_opening_surprising_effectiveness": "IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot",
    "bridge": "lerobot/bridge",
    "berkeley_cable_routing": "lerobot/berkeley_cable_routing",
    "berkeley_autolab_ur5": "lerobot/berkeley_autolab_ur5",
}


def download_hf_dataset(hf_repo: str, output_dir: Path):
    """Download dataset from HuggingFace Hub."""

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("HuggingFace datasets library not installed")
        logger.error("Install with: pip install datasets")
        return False

    logger.info(f"Downloading from HuggingFace: {hf_repo}")
    logger.info(f"Saving to: {output_dir}")
    logger.info("This may take a while depending on dataset size...")

    try:
        # Download and save dataset
        dataset = load_dataset(hf_repo, split="train")

        logger.info(f"Dataset loaded: {len(dataset)} frames")
        logger.info(f"Saving to disk at {output_dir}...")

        # Create parent directory
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Save to disk
        dataset.save_to_disk(str(output_dir))

        logger.info(f"✓ Dataset saved successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def download_with_snapshot(hf_repo: str, output_dir: Path):
    """Alternative: Download using huggingface_hub snapshot_download."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub library not installed")
        logger.error("Install with: pip install huggingface-hub")
        return False

    logger.info(f"Downloading snapshot from HuggingFace: {hf_repo}")
    logger.info(f"Saving to: {output_dir}")
    logger.info("This may take a while depending on dataset size...")

    try:
        # Download all files
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )

        logger.info(f"✓ Dataset downloaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-converted LeRobot datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for datasets",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=list(HF_DATASET_REPOS.keys()),
        help="Name of the dataset to download",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["datasets", "snapshot"],
        default="datasets",
        help="Download method: 'datasets' (load_dataset) or 'snapshot' (raw files)",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="Custom HuggingFace repo (overrides default mapping)",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = base_dir / "hf" / args.dataset_name

    # Determine HF repo
    if args.hf_repo:
        hf_repo = args.hf_repo
    else:
        hf_repo = HF_DATASET_REPOS[args.dataset_name]

    logger.info("=" * 80)
    logger.info(f"DOWNLOADING DATASET: {args.dataset_name}")
    logger.info("=" * 80)
    logger.info(f"HuggingFace repo: {hf_repo}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Download method: {args.method}")
    logger.info("=" * 80)

    # Check if already exists
    if output_dir.exists():
        response = input(f"\n⚠ Directory already exists: {output_dir}\nOverwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Download cancelled")
            sys.exit(0)

    # Download
    if args.method == "datasets":
        success = download_hf_dataset(hf_repo, output_dir)
    else:
        success = download_with_snapshot(hf_repo, output_dir)

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ DOWNLOAD COMPLETED SUCCESSFULLY")
        logger.info(f"Dataset location: {output_dir}")

        # Show dataset structure
        if output_dir.exists():
            files = list(output_dir.iterdir())
            logger.info(f"\nFiles in dataset directory ({len(files)} items):")
            for f in sorted(files)[:10]:  # Show first 10
                logger.info(f"  - {f.name}")
            if len(files) > 10:
                logger.info(f"  ... and {len(files) - 10} more")
    else:
        logger.error("✗ DOWNLOAD FAILED")
    logger.info("=" * 80)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
