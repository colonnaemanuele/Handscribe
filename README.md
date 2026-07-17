<!-- ## Features

- **SlowFast Network Integration**: Implements SlowFast models for video-based CSLR.
- **Dataset Support**: Includes preprocessing scripts for datasets like CSL, CSL-Daily, LIS, and PHOENIX-2014-T.
- **Evaluation Tools**: Provides Word Error Rate (WER) calculation for model evaluation.
- **Customizable Training**: Supports various configurations for training and testing.

## Repository Structure

- **`main.py`**: Entry point for training, testing, and feature extraction.
- **`seq_scripts.py`**: Contains functions for training, evaluation, and feature generation.
- **`hanscribe/`**: Main module directory containing:
  - `configs/`: Configuration files for datasets and models.
  - `dataset/`: Data loaders for different datasets.
  - `preprocess/`: Scripts for dataset preprocessing.
  - `slowfast_modules/`: Implementation of SlowFast models and utilities.
  - `evaluation/`: Tools for evaluating model performance.
  - `utils/`: Utility functions for logging, parameter parsing, and more.
- **`ckpt/`**: Directory for storing pretrained model checkpoints.
- **`work_dir/`**: Directory for storing training outputs and logs. -->


# Handscribe

Handscribe is a framework for Continuous Sign Language Translation and Recognition (CSLTR) using SlowFast networks. It includes tools for dataset preprocessing, model training, and performance evaluation.

## Prerequisites

- Python 3.12
- PyTorch 2.6.0
- Torchvision 0.21.0
- Other dependencies listed in `Handscribe/pyproject.toml`

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
Install the project dependencies from `pyproject.toml`:
```bash
uv sync
```

or

```bash
pip install . # inside the folder contains pyproject.toml
```

## Before Running Anything

> [!IMPORTANT]
> ⚠️ Important: Complete the steps below before starting training.

Please create the following directory and download the `.pkl` file for the SlowFast model:

```
mkdir ckpt && cd ckpt
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl
```

<!-- Additionally, you need to clone the `ctcdecode` repository in order to run training:

```
git clone --recursive https://github.com/WayenVan/ctcdecode.git
cd ctcdecode && pip install .
``` -->

You’re now ready to go! 🚀

## Dataset Preparation
Preprocessing scripts for supported datasets are in `hanscribe/preprocess/`. Examples:
- `dataset_preprocess-CSL.py`
- `dataset_preprocess-CSL-Daily.py`
- `dataset_preprocess-lis.py` ???? MAGER!
- `dataset_preprocess-T.py`

## Data Preparation
Please follow the instruction in [CorrNet](https://github.com/hulianyuyy/CorrNet) github repo to download and preprocess the datasets (PHOENIX2014, PHOENIX2014-T, CSL-Daily).
The structure of dataset directory is as follows (There may be other additional directories.):
```
dataset
├── phoenix2014
│   └── phoenix-2014-multisigner
│       └── features
│           └── fullFrame-256x256px
├── phoenix2014-T
│   └── features
│       └── fullFrame-256x256px
└── CSL-Daily
    └── sentence
        └── frames_256x256
```

## Training

To train a model, use one of the following commands:

### Using `uv` (recommended):
```bash
uv run main.py --dataset phoenix2014-T --loss-weights Slow=0.25 Fast=0.25 --work-dir ./work_dir/phoenix2014T/
```
### Using `python` (if `uv` is not installed):
```bash
python main.py --dataset phoenix2014-T --loss-weights Slow=0.25 Fast=0.25 --work-dir ./work_dir/phoenix2014T/
```
