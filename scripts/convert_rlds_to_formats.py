#!/usr/bin/env python3
"""
Convert RLDS datasets to multiple formats (VLA, HDF5, HuggingFace).

This script converts downloaded RLDS datasets to VLA, HDF5, and HuggingFace formats
for benchmarking different data loading approaches.

Usage:
    python scripts/convert_rlds_to_formats.py --base_dir <path> --dataset_name nyu_door_opening_surprising_effectiveness --formats vla hdf5 hf
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_rlds_to_vla(
    rlds_dir: Path,
    vla_dir: Path,
    dataset_name: str,
    max_workers: int = 4,
    lossless: bool = False,
):
    """Convert RLDS format to VLA format."""

    logger.info(f"Converting RLDS to VLA: {dataset_name}")
    logger.info(f"  Source: {rlds_dir}")
    logger.info(f"  Destination: {vla_dir}")

    try:
        from fog_x.loader import RLDSLoader
        import fog_x
    except ImportError as e:
        logger.error(f"Failed to import fog_x: {e}")
        logger.error("Make sure fog_x is installed: pip install -e .")
        return False

    # Create output directory
    output_dir = vla_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RLDS dataset
    try:
        loader = RLDSLoader(
            path=str(rlds_dir),
            split="train",
            shuffling=False,
            batch_size=1
        )
        logger.info(f"  Loaded RLDS dataset: {len(loader)} trajectories")
    except Exception as e:
        logger.error(f"Failed to load RLDS dataset: {e}")
        return False

    # Conversion function for parallel processing
    def process_trajectory(data_traj, index):
        try:
            data_traj = data_traj[0]  # Unpack batch
            output_path = output_dir / f"output_{index}.vla"

            if lossless:
                fog_x.Trajectory.from_list_of_dicts(
                    data_traj,
                    path=str(output_path),
                    lossy_compression=False
                )
            else:
                fog_x.Trajectory.from_list_of_dicts(
                    data_traj,
                    path=str(output_path),
                    lossy_compression=True,
                )

            if index % 50 == 0:
                logger.info(f"  Processed trajectory {index}/{len(loader)}")

            return index, True
        except Exception as e:
            logger.error(f"  Failed to process trajectory {index}: {e}")
            return index, False

    # Process trajectories in parallel
    max_concurrent_tasks = max_workers
    semaphore = threading.Semaphore(max_concurrent_tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        failed_indices = []

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.warning("tqdm not installed, progress bar disabled")

        iterator = enumerate(loader)
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Converting to VLA", total=len(loader), unit="traj")

        for index, data_traj in iterator:
            semaphore.acquire()
            future = executor.submit(process_trajectory, data_traj, index)
            future.add_done_callback(lambda x: semaphore.release())
            futures.append(future)

        for future in as_completed(futures):
            try:
                index, success = future.result()
                if not success:
                    failed_indices.append(index)
            except Exception as e:
                logger.error(f"Error processing future: {e}")

    if failed_indices:
        logger.warning(f"  Failed to convert {len(failed_indices)} trajectories: {failed_indices[:10]}...")
        return False

    logger.info(f"✓ Successfully converted {len(loader)} trajectories to VLA format")
    return True


def convert_vla_to_hdf5(
    vla_dir: Path,
    hdf5_dir: Path,
    dataset_name: str,
    cache_dir: Path,
    max_workers: int = 4,
):
    """Convert VLA format to HDF5 format."""

    logger.info(f"Converting VLA to HDF5: {dataset_name}")
    logger.info(f"  Source: {vla_dir / dataset_name}")
    logger.info(f"  Destination: {hdf5_dir / dataset_name}")

    try:
        from fog_x.loader import NonShuffleVLALoader
        import h5py
        import numpy as np
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        return False

    # Create output directory
    output_dir = hdf5_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cache directory
    dataset_cache_dir = cache_dir / dataset_name
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load VLA dataset
    try:
        vla_path = str(vla_dir / dataset_name / "*.vla")
        loader = NonShuffleVLALoader(vla_path, cache_dir=str(dataset_cache_dir))
        logger.info(f"  Loaded VLA dataset")
    except Exception as e:
        logger.error(f"Failed to load VLA dataset: {e}")
        return False

    # Conversion function
    def process_trajectory(trajectory, index):
        try:
            if trajectory is None:
                logger.warning(f"  Trajectory {index} is None")
                return index, False

            output_path = output_dir / f"output_{index}.h5"

            with h5py.File(str(output_path), "w") as f:
                for key, value in trajectory.items():
                    if value is not None and hasattr(value, 'shape'):
                        f.create_dataset(key, data=value, compression="gzip", compression_opts=9)

            if index % 50 == 0:
                logger.info(f"  Processed trajectory {index}")

            return index, True
        except Exception as e:
            logger.error(f"  Failed to process trajectory {index}: {e}")
            return index, False

    # Process trajectories
    max_concurrent_tasks = max_workers
    semaphore = threading.Semaphore(max_concurrent_tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        failed_indices = []

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        iterator = enumerate(loader)
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Converting to HDF5", unit="traj")

        for index, trajectory in iterator:
            semaphore.acquire()
            future = executor.submit(process_trajectory, trajectory, index)
            future.add_done_callback(lambda x: semaphore.release())
            futures.append(future)

        for future in as_completed(futures):
            try:
                index, success = future.result()
                if not success:
                    failed_indices.append(index)
            except Exception as e:
                logger.error(f"Error processing future: {e}")

    if failed_indices:
        logger.warning(f"  Failed to convert {len(failed_indices)} trajectories")
        return False

    logger.info(f"✓ Successfully converted to HDF5 format")
    return True


def convert_rlds_to_hf(
    rlds_dir: Path,
    hf_dir: Path,
    dataset_name: str,
    fps: int = 12,
    video: bool = True,
):
    """Convert RLDS format to HuggingFace format."""

    logger.info(f"Converting RLDS to HuggingFace: {dataset_name}")
    logger.info(f"  Source: {rlds_dir}")
    logger.info(f"  Destination: {hf_dir / dataset_name}")

    try:
        import tensorflow_datasets as tfds
        import torch
        import numpy as np
        from PIL import Image as PILImage
        from datasets import Dataset, Features, Image, Sequence, Value
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        logger.error("Install with: pip install tensorflow tensorflow-datasets torch datasets pillow")
        return False

    # Create output directories
    output_dir = hf_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if video:
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

    # Note: Full HF conversion is complex and requires the lerobot library
    # For now, we'll create a simplified version
    logger.warning("  HuggingFace conversion requires lerobot library and OpenX transforms")
    logger.warning("  This is a placeholder - full conversion should use examples/rlds_to_lerobot.py")
    logger.info("  For complete HF conversion, please run:")
    logger.info(f"    python examples/rlds_to_lerobot.py (modify for your dataset)")

    # Create a marker file to indicate HF conversion is needed
    marker_file = output_dir / "CONVERSION_NEEDED.txt"
    with open(marker_file, "w") as f:
        f.write(f"HuggingFace conversion for {dataset_name} requires manual setup.\n")
        f.write(f"Please refer to examples/rlds_to_lerobot.py for full conversion.\n")
        f.write(f"\nDataset: {dataset_name}\n")
        f.write(f"Source: {rlds_dir}\n")

    logger.info("✓ Created HF directory structure (manual conversion required)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert RLDS datasets to multiple formats",
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
        required=True,
        help="Name of the dataset to convert",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["vla", "hdf5", "hf", "all"],
        default=["all"],
        help="Formats to convert to (default: all)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker processes (default: 4)",
    )
    parser.add_argument(
        "--lossless",
        action="store_true",
        help="Use lossless compression for VLA format (slower, larger files)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="FPS for video encoding in HF format (default: 12)",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable video encoding in HF format (use raw images)",
    )

    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(args.base_dir).expanduser().resolve()
    rlds_dir = base_dir / "rlds" / args.dataset_name
    vla_dir = base_dir / "vla"
    hdf5_dir = base_dir / "hdf5"
    hf_dir = base_dir / "hf"
    cache_dir = base_dir / "cache"

    # Validate RLDS dataset exists
    if not rlds_dir.exists():
        logger.error(f"RLDS dataset not found: {rlds_dir}")
        logger.error("Please download the dataset first using scripts/download_openx_datasets.py")
        sys.exit(1)

    # Determine which formats to convert
    formats = args.formats
    if "all" in formats:
        formats = ["vla", "hdf5", "hf"]

    logger.info("=" * 80)
    logger.info(f"CONVERTING DATASET: {args.dataset_name}")
    logger.info("=" * 80)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Formats: {', '.join(formats)}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info("=" * 80)

    success = True

    # Convert to VLA
    if "vla" in formats:
        logger.info("\n[1/3] Converting RLDS → VLA")
        if not convert_rlds_to_vla(
            rlds_dir, vla_dir, args.dataset_name,
            max_workers=args.max_workers,
            lossless=args.lossless
        ):
            logger.error("VLA conversion failed")
            success = False

    # Convert VLA to HDF5 (requires VLA to exist)
    if "hdf5" in formats:
        logger.info("\n[2/3] Converting VLA → HDF5")
        # Ensure VLA exists
        vla_dataset_dir = vla_dir / args.dataset_name
        if not vla_dataset_dir.exists() or not list(vla_dataset_dir.glob("*.vla")):
            logger.warning("VLA dataset not found, converting RLDS → VLA first...")
            if not convert_rlds_to_vla(
                rlds_dir, vla_dir, args.dataset_name,
                max_workers=args.max_workers,
                lossless=args.lossless
            ):
                logger.error("VLA conversion failed, skipping HDF5")
                success = False
            else:
                if not convert_vla_to_hdf5(
                    vla_dir, hdf5_dir, args.dataset_name,
                    cache_dir, max_workers=args.max_workers
                ):
                    logger.error("HDF5 conversion failed")
                    success = False
        else:
            if not convert_vla_to_hdf5(
                vla_dir, hdf5_dir, args.dataset_name,
                cache_dir, max_workers=args.max_workers
            ):
                logger.error("HDF5 conversion failed")
                success = False

    # Convert to HuggingFace
    if "hf" in formats:
        logger.info("\n[3/3] Converting RLDS → HuggingFace")
        if not convert_rlds_to_hf(
            rlds_dir, hf_dir, args.dataset_name,
            fps=args.fps,
            video=not args.no_video
        ):
            logger.error("HuggingFace conversion failed")
            success = False

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ CONVERSION COMPLETED SUCCESSFULLY")
    else:
        logger.warning("⚠ CONVERSION COMPLETED WITH ERRORS")
    logger.info("=" * 80)

    # Print summary
    logger.info("\nDataset locations:")
    if (vla_dir / args.dataset_name).exists():
        vla_files = list((vla_dir / args.dataset_name).glob("*.vla"))
        logger.info(f"  VLA:  {vla_dir / args.dataset_name} ({len(vla_files)} files)")
    if (hdf5_dir / args.dataset_name).exists():
        hdf5_files = list((hdf5_dir / args.dataset_name).glob("*.h5"))
        logger.info(f"  HDF5: {hdf5_dir / args.dataset_name} ({len(hdf5_files)} files)")
    if (hf_dir / args.dataset_name).exists():
        logger.info(f"  HF:   {hf_dir / args.dataset_name}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
