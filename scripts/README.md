# OpenX Dataset Preparation for Benchmarking

This directory contains scripts to download, convert, and prepare OpenX datasets for benchmarking different data loading formats.

## Overview

The benchmarking pipeline consists of three main stages:

1. **Download**: Get RLDS datasets from TensorFlow Datasets
2. **Convert**: Transform RLDS to VLA, HDF5, and HuggingFace formats
3. **Benchmark**: Measure loading performance across formats

## Quick Start

### 1. Setup Directory Structure

Choose a base directory for your datasets (e.g., `~/robodm/data` or `/data/fog_x`):

```bash
# Create directory structure
python scripts/download_openx_datasets.py --base_dir ~/robodm/data --setup
```

This creates:
```
~/robodm/data/
├── rlds/      # Original RLDS format (downloaded)
├── vla/       # Converted VLA format
├── hdf5/      # Converted HDF5 format
├── hf/        # Converted HuggingFace format
└── cache/     # Temporary cache
```

### 2. Download RLDS Dataset

Start with the smallest dataset for testing:

```bash
# Get download instructions for nyu_door dataset (435 trajectories)
python scripts/download_openx_datasets.py \
    --base_dir ~/robodm/data \
    --dataset_name nyu_door_opening_surprising_effectiveness
```

This will show you three download methods. **Recommended: Method 1** (using tfds):

```python
import tensorflow_datasets as tfds
import os

# Set download directory
os.environ['TFDS_DATA_DIR'] = '~/robodm/data/rlds'

# Download the dataset
builder = tfds.builder('nyu_door_opening_surprising_effectiveness')
builder.download_and_prepare()
```

### 3. Verify Download

```bash
python scripts/download_openx_datasets.py \
    --base_dir ~/robodm/data \
    --dataset_name nyu_door_opening_surprising_effectiveness \
    --verify
```

### 4. Convert to All Formats

```bash
# Automated conversion to all formats (VLA, HDF5, HF)
python scripts/prepare_benchmark_data.py \
    --base_dir ~/robodm/data \
    --dataset_name nyu_door_opening_surprising_effectiveness \
    --auto
```

Or convert to specific formats:

```bash
# Convert to VLA and HDF5 only
python scripts/convert_rlds_to_formats.py \
    --base_dir ~/robodm/data \
    --dataset_name nyu_door_opening_surprising_effectiveness \
    --formats vla hdf5 \
    --max_workers 4
```

### 5. Run Benchmark

```bash
# Benchmark all available formats
python benchmarks/openx.py \
    --exp_dir ~/robodm/data \
    --dataset_names nyu_door_opening_surprising_effectiveness \
    --num_batches 100 \
    --batch_size 16
```

Or benchmark specific formats:

```bash
# Benchmark only VLA and HDF5
python benchmarks/openx.py \
    --exp_dir ~/robodm/data \
    --dataset_names nyu_door_opening_surprising_effectiveness \
    --num_batches 100 \
    --batch_size 16 \
    --formats vla hdf5
```

## Available Datasets

| Dataset Name | Trajectories | Estimated Size | Description |
|-------------|--------------|----------------|-------------|
| `nyu_door_opening_surprising_effectiveness` | 435 | ~10 GB | NYU Door Opening (recommended for testing) |
| `berkeley_autolab_ur5` | 896 | ~15 GB | Berkeley AUTOLab UR5 |
| `berkeley_cable_routing` | 1,482 | ~20 GB | UC Berkeley Cable Routing |
| `bridge` | 25,460 | ~100 GB | Bridge Dataset v2 (very large!) |

**Recommendation**: Start with `nyu_door_opening_surprising_effectiveness` to test the pipeline before processing larger datasets.

## Detailed Usage

### Script 1: `download_openx_datasets.py`

Provides download instructions and verification for RLDS datasets.

```bash
# List all available datasets
python scripts/download_openx_datasets.py --list

# Get download instructions
python scripts/download_openx_datasets.py \
    --base_dir <path> \
    --dataset_name <dataset>

# Verify downloaded dataset
python scripts/download_openx_datasets.py \
    --base_dir <path> \
    --dataset_name <dataset> \
    --verify
```

### Script 2: `convert_rlds_to_formats.py`

Converts RLDS datasets to other formats.

```bash
python scripts/convert_rlds_to_formats.py \
    --base_dir <path> \
    --dataset_name <dataset> \
    --formats vla hdf5 hf \
    --max_workers 4 \
    [--lossless]  # Use lossless compression for VLA (slower, larger)
```

**Conversion Pipeline**:
- RLDS → VLA: Direct conversion using `fog_x.Trajectory`
- VLA → HDF5: Converts from VLA format
- RLDS → HF: Direct conversion (requires lerobot library)

**Note**: HuggingFace conversion currently creates directory structure only. For full HF conversion with video encoding, use `examples/rlds_to_lerobot.py` as a reference.

### Script 3: `prepare_benchmark_data.py`

Master orchestration script that handles the entire pipeline.

```bash
# Interactive mode - guides you through the process
python scripts/prepare_benchmark_data.py \
    --base_dir <path> \
    --dataset_name <dataset>

# Automated mode - runs all conversions
python scripts/prepare_benchmark_data.py \
    --base_dir <path> \
    --dataset_name <dataset> \
    --auto

# Check status only
python scripts/prepare_benchmark_data.py \
    --base_dir <path> \
    --dataset_name <dataset> \
    --status_only

# Prepare and benchmark in one command
python scripts/prepare_benchmark_data.py \
    --base_dir <path> \
    --dataset_name <dataset> \
    --auto \
    --benchmark \
    --num_batches 100
```

### Script 4: `benchmarks/openx.py`

Benchmarks data loading performance across formats.

```bash
python benchmarks/openx.py \
    --exp_dir <base_dir> \
    --dataset_names <dataset1> [<dataset2> ...] \
    --num_batches 1000 \
    --batch_size 16 \
    --log_frequency 20 \
    [--formats vla hdf5 hf rlds]  # Optional: specify formats to benchmark
```

**Output**: Creates CSV files with timing results:
- `<dataset_name>_results.csv` - Per-batch loading times
- `format_comparison_results.csv` - Aggregated results across formats

## Complete Workflow Example

Here's a complete workflow for benchmarking the NYU Door dataset:

```bash
# 1. Setup
export BASE_DIR=~/robodm/data
export DATASET=nyu_door_opening_surprising_effectiveness

# 2. Create directory structure
python scripts/download_openx_datasets.py --base_dir $BASE_DIR --setup

# 3. Download RLDS dataset (use one of the methods shown)
python scripts/download_openx_datasets.py --base_dir $BASE_DIR --dataset_name $DATASET
# Follow the instructions to download using tfds

# 4. Verify download
python scripts/download_openx_datasets.py --base_dir $BASE_DIR --dataset_name $DATASET --verify

# 5. Convert to all formats (VLA, HDF5, HF) and run benchmark
python scripts/prepare_benchmark_data.py \
    --base_dir $BASE_DIR \
    --dataset_name $DATASET \
    --auto \
    --benchmark \
    --num_batches 100 \
    --batch_size 16

# Results will be saved to CSV files in the current directory
```

## Scaling to Multiple Datasets

Once you've tested with the smallest dataset, you can process all four datasets:

```bash
#!/bin/bash
BASE_DIR=~/robodm/data
DATASETS=(
    "nyu_door_opening_surprising_effectiveness"
    "berkeley_autolab_ur5"
    "berkeley_cable_routing"
    "bridge"
)

for dataset in "${DATASETS[@]}"; do
    echo "Processing $dataset..."

    # Download instructions
    python scripts/download_openx_datasets.py --base_dir $BASE_DIR --dataset_name $dataset

    # Wait for manual download
    read -p "Press enter when $dataset is downloaded..."

    # Verify
    python scripts/download_openx_datasets.py --base_dir $BASE_DIR --dataset_name $dataset --verify

    # Convert
    python scripts/prepare_benchmark_data.py --base_dir $BASE_DIR --dataset_name $dataset --auto

    # Benchmark
    python benchmarks/openx.py --exp_dir $BASE_DIR --dataset_names $dataset --num_batches 1000 --batch_size 16
done
```

## Performance Tips

1. **Parallel Processing**: Use `--max_workers` to control parallelism during conversion
   - More workers = faster conversion, but higher memory usage
   - Recommended: 4-8 workers depending on your system

2. **VLA Compression**: Use lossy compression (default) for smaller files and faster loading
   - Use `--lossless` only if you need exact data preservation

3. **Batch Size**: Larger batch sizes improve throughput but require more memory
   - Start with batch_size=16, increase if memory allows

4. **Testing**: Always test with `nyu_door_opening_surprising_effectiveness` first
   - Smallest dataset (435 trajectories)
   - Faster to download and convert
   - Good for validating the pipeline

## Troubleshooting

### Issue: "RLDS dataset not found"
- Make sure you've downloaded the dataset using one of the three methods
- Verify the dataset exists: `ls $BASE_DIR/rlds/<dataset_name>/`
- Check for .tfrecord files in the directory

### Issue: "VLA conversion failed"
- Ensure fog_x is installed: `pip install -e .`
- Check available disk space
- Try reducing `--max_workers` to avoid memory issues

### Issue: "HDF5 conversion failed"
- VLA dataset must exist first
- Check `h5py` is installed: `pip install h5py`
- Verify VLA files: `ls $BASE_DIR/vla/<dataset_name>/*.vla`

### Issue: "Benchmark finds no datasets"
- Verify `--exp_dir` matches your base directory
- Check that converted formats exist: `ls $BASE_DIR/{vla,hdf5,hf}/<dataset_name>/`
- Use `--status_only` to check what's available

### Issue: "Out of memory during conversion"
- Reduce `--max_workers` (try 2 or 1)
- Process smaller datasets first
- Close other applications

## File Formats Explained

- **RLDS**: TensorFlow Records format, hierarchical structure with episodes and steps
- **VLA**: Custom format from this repo, optimized for video data with AV1 compression
- **HDF5**: Hierarchical Data Format 5, widely used in scientific computing
- **HuggingFace**: HuggingFace Datasets format, compatible with LeRobot library

## Requirements

```bash
# Core dependencies
pip install tensorflow tensorflow-datasets
pip install torch torchvision
pip install h5py
pip install datasets pillow
pip install pandas numpy
pip install tqdm

# fog_x library (this repo)
pip install -e .

# Optional: for full HF conversion with video
pip install lerobot
```

## References

- Bridge Dataset: https://rail-berkeley.github.io/bridgedata/
- OpenX Embodiment: https://robotics-transformer-x.github.io/
- TensorFlow Datasets: https://www.tensorflow.org/datasets
- LeRobot: https://github.com/huggingface/lerobot
