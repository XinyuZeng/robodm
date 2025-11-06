#!/usr/bin/env python3
"""
Master orchestration script for preparing OpenX benchmark datasets.

This script orchestrates the entire data preparation pipeline:
1. Verify RLDS dataset is downloaded
2. Convert to required formats (VLA, HDF5, HuggingFace)
3. Prepare for benchmarking

Usage:
    # Interactive mode - guides you through the process
    python scripts/prepare_benchmark_data.py --base_dir <path> --dataset_name nyu_door_opening_surprising_effectiveness

    # Automated mode - runs all conversions
    python scripts/prepare_benchmark_data.py --base_dir <path> --dataset_name nyu_door_opening_surprising_effectiveness --auto
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATASET_INFO = {
    "bridge": {
        "num_trajectories": 25460,
        "description": "Bridge Dataset v2",
    },
    "nyu_door_opening_surprising_effectiveness": {
        "num_trajectories": 435,
        "description": "NYU Door Opening",
    },
    "berkeley_cable_routing": {
        "num_trajectories": 1482,
        "description": "Berkeley Cable Routing",
    },
    "berkeley_autolab_ur5": {
        "num_trajectories": 896,
        "description": "Berkeley AUTOLab UR5",
    },
}


def verify_dataset_downloaded(base_dir: Path, dataset_name: str) -> bool:
    """Verify that RLDS dataset exists."""

    rlds_dir = base_dir / "rlds" / dataset_name

    if not rlds_dir.exists():
        return False

    # Check for tfrecord files
    tfrecord_files = list(rlds_dir.glob("**/*.tfrecord*"))
    if not tfrecord_files:
        return False

    # Try to load with tfds
    try:
        import tensorflow_datasets as tfds
        builder = tfds.builder_from_directory(str(rlds_dir))
        return True
    except Exception as e:
        logger.warning(f"Dataset exists but cannot be loaded: {e}")
        return False


def check_format_exists(base_dir: Path, dataset_name: str, format_name: str) -> bool:
    """Check if a specific format has been converted."""

    format_dir = base_dir / format_name / dataset_name

    if not format_dir.exists():
        return False

    # Check for files
    if format_name == "vla":
        return len(list(format_dir.glob("*.vla"))) > 0
    elif format_name == "hdf5":
        return len(list(format_dir.glob("*.h5"))) > 0
    elif format_name == "hf":
        return format_dir.exists()  # HF is a directory structure

    return False


def print_status(base_dir: Path, dataset_name: str):
    """Print current status of dataset preparation."""

    logger.info("\n" + "=" * 80)
    logger.info(f"DATASET PREPARATION STATUS: {dataset_name}")
    logger.info("=" * 80)

    # Check RLDS
    rlds_exists = verify_dataset_downloaded(base_dir, dataset_name)
    status_rlds = "✓ Downloaded" if rlds_exists else "✗ Not found"
    logger.info(f"RLDS (source):  {status_rlds}")

    if not rlds_exists:
        logger.info("\n→ Next step: Download RLDS dataset")
        logger.info(f"  Run: python scripts/download_openx_datasets.py --base_dir {base_dir} --dataset_name {dataset_name}")
        return False

    # Check converted formats
    formats_status = {}
    for fmt in ["vla", "hdf5", "hf"]:
        exists = check_format_exists(base_dir, dataset_name, fmt)
        formats_status[fmt] = exists
        status = "✓ Converted" if exists else "✗ Not converted"
        logger.info(f"{fmt.upper():12s}  {status}")

    all_converted = all(formats_status.values())

    if all_converted:
        logger.info("\n✓ All formats ready for benchmarking!")
        return True
    else:
        missing = [fmt for fmt, exists in formats_status.items() if not exists]
        logger.info(f"\n→ Next step: Convert to missing formats: {', '.join(missing)}")
        logger.info(f"  Run: python scripts/convert_rlds_to_formats.py --base_dir {base_dir} --dataset_name {dataset_name} --formats {' '.join(missing)}")
        return False


def run_download_instructions(base_dir: Path, dataset_name: str):
    """Show download instructions."""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DOWNLOAD RLDS DATASET")
    logger.info("=" * 80)

    cmd = [
        "python", "scripts/download_openx_datasets.py",
        "--base_dir", str(base_dir),
        "--dataset_name", dataset_name
    ]

    subprocess.run(cmd)


def run_conversion(base_dir: Path, dataset_name: str, formats: list[str], max_workers: int = 4):
    """Run format conversion."""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CONVERT TO REQUIRED FORMATS")
    logger.info("=" * 80)

    cmd = [
        "python", "scripts/convert_rlds_to_formats.py",
        "--base_dir", str(base_dir),
        "--dataset_name", dataset_name,
        "--formats", *formats,
        "--max_workers", str(max_workers)
    ]

    logger.info(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_benchmark(base_dir: Path, dataset_name: str, num_batches: int = 100, batch_size: int = 16):
    """Run benchmark script."""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: RUN BENCHMARK")
    logger.info("=" * 80)

    cmd = [
        "python", "benchmarks/openx.py",
        "--exp_dir", str(base_dir),
        "--dataset_names", dataset_name,
        "--num_batches", str(num_batches),
        "--batch_size", str(batch_size)
    ]

    logger.info(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Master script for preparing OpenX benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory for dataset storage",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=list(DATASET_INFO.keys()),
        required=True,
        help="Name of the dataset to prepare",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatic mode - run all conversions without prompts",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["vla", "hdf5", "hf", "all"],
        default=["all"],
        help="Formats to convert (default: all)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker processes for conversion (default: 4)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after preparation",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of batches for benchmarking (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for benchmarking (default: 16)",
    )
    parser.add_argument(
        "--status_only",
        action="store_true",
        help="Only print status and exit",
    )

    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(args.base_dir).expanduser().resolve()

    # Determine formats
    formats = args.formats
    if "all" in formats:
        formats = ["vla", "hdf5", "hf"]

    # Print header
    logger.info("\n" + "=" * 80)
    logger.info("OPENX BENCHMARK DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Target formats: {', '.join(formats)}")
    logger.info("=" * 80)

    # Check status
    status_ok = print_status(base_dir, args.dataset_name)

    if args.status_only:
        sys.exit(0 if status_ok else 1)

    # If dataset not downloaded, show instructions
    if not verify_dataset_downloaded(base_dir, args.dataset_name):
        logger.info("\n⚠ RLDS dataset not found!")
        run_download_instructions(base_dir, args.dataset_name)
        logger.info("\n" + "=" * 80)
        logger.info("Please download the dataset first, then run this script again.")
        logger.info("=" * 80)
        sys.exit(1)

    # Check which formats need conversion
    missing_formats = []
    for fmt in formats:
        if not check_format_exists(base_dir, args.dataset_name, fmt):
            missing_formats.append(fmt)

    if not missing_formats:
        logger.info("\n✓ All requested formats are already available!")
        if args.benchmark:
            run_benchmark(base_dir, args.dataset_name, args.num_batches, args.batch_size)
        sys.exit(0)

    # Run conversion
    if args.auto:
        logger.info(f"\n→ Converting to: {', '.join(missing_formats)}")
        success = run_conversion(base_dir, args.dataset_name, missing_formats, args.max_workers)

        if not success:
            logger.error("\n✗ Conversion failed!")
            sys.exit(1)

        logger.info("\n✓ Conversion completed successfully!")

        # Print final status
        print_status(base_dir, args.dataset_name)

        # Run benchmark if requested
        if args.benchmark:
            run_benchmark(base_dir, args.dataset_name, args.num_batches, args.batch_size)

    else:
        # Interactive mode
        logger.info(f"\nMissing formats: {', '.join(missing_formats)}")
        response = input("\nDo you want to convert them now? [y/N]: ")

        if response.lower() in ['y', 'yes']:
            success = run_conversion(base_dir, args.dataset_name, missing_formats, args.max_workers)

            if success:
                logger.info("\n✓ Conversion completed successfully!")
                print_status(base_dir, args.dataset_name)

                if args.benchmark:
                    response = input("\nDo you want to run the benchmark now? [y/N]: ")
                    if response.lower() in ['y', 'yes']:
                        run_benchmark(base_dir, args.dataset_name, args.num_batches, args.batch_size)
            else:
                logger.error("\n✗ Conversion failed!")
                sys.exit(1)
        else:
            logger.info("\nConversion skipped. Run with --auto to convert automatically.")


if __name__ == "__main__":
    main()
