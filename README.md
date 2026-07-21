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
└── phoenix2014-T
    └── features
        └── fullFrame-256x256px
```

## Download Fieldwork dataset

Run the notebook located at `Handscribe/gloss_generator/download_fieldwork.ipynb`.

This will download the [Wav2Gloss Fieldwork dataset from HuggingFace](https://huggingface.co/datasets/wav2gloss/fieldwork), resulting in the following directory structure:

```
fieldwork_data
├── seen
│ ├── ainu1240.csv
│ ├── apah1238.csv
│ ├── arap1274.csv
│ └── ...
│
├── unseen
│ ├── arta1239.csv
│ ├── balk1252.csv
│ ├── kach1280.csv
│ └── ...
│
├── testfull.csv
├── testseen_lang.csv
├── test\_\_unseen_lang.csv
└── train.csv
```

# SLT Module

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

# Gloss Generator Module

> [!IMPORTANT]
> 📢 By this point, you should have downloaded either the PHOENIX-2014-T dataset or the Fieldwork dataset. If not, please complete that step before continuing.

## Fine-Tuning

> [!NOTE]
> 📌 You can pick any LLM available in [Unsloth&#39;s HuggingFace repository](https://huggingface.co/unsloth).

To fine-tune an LLM, use the following command:

```
python Handscribe/recognizer/unsloth_finetuning.py --llm_to_tune UNSLOTH_HF_LLM_ID --data_csv_path PATH_TO_DATASET_CSV --out_dir OUTPUT_DIR --max_seq_length MAX_SEQ_LENGTH
```

- `UNSLOTH_HF_LLM_ID`: ID of the LLM in Unsloth's HuggingFace repository (must be listed in `AVAILABLE_MODELS`).
- `PATH_TO_DATASET_CSV`: path to a CSV file whose formatting matches `Handscribe/dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv` or `fieldwork_data/train.csv`.
- `OUTPUT_DIR`: folder where the fine-tuned model and tokenizer will be saved, and should also include an identifier for the fine-tuned model (e.g., `model123_finetuned`), since it is what you'll later pass to `--llm_to_use` during inference.
- `MAX_SEQ_LENGTH`: the maximum sequence length the LLM will generate (e.g., `2048`).

> [!TIP]
> ❓ To test on a subset of the dataset, add `--use_data_subset --subset_amount AMOUNT` (e.g., `1000`). If VRAM is a problem, use `--load_4bit` to load the LLM at 4-bit quantization.

Example: fine-tuning `unsloth/Llama-3.2-3B-Instruct` on the Phoenix dataset:

```
python Handscribe/recognizer/unsloth_finetuning.py --llm_to_tune unsloth/Llama-3.2-3B-Instruct --data_csv_path Handscribe/dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv --out_dir ft_output/gloss_llama32_1k --max_seq_length 2048
```

## Inference with a Fine-Tuned Model

To run inference with a previously fine-tuned LLM, use the following command:

```
python Handscribe/recognizer/tuned_llms_inference.py --llm_to_use FINETUNED_MODEL_IDENTIFIER --data_csv_path TEST_DATA_CSV_PATH --sentences_to_test NO_SENTS_TO_TEST --max_seq_length MAX_SEQ_LENGTH --data_language DATA_LANGUAGE
```

- `FINETUNED_MODEL_IDENTIFIER`: the identifier assigned to the model during fine-tuning (the last path component of the value you passed to `--out_dir`, e.g. `gloss_llama32_1k`). Must be one of the models listed in `TUNED_MODELS`.
- `TEST_DATA_CSV_PATH`: path to a CSV file matching either the PHOENIX-2014-T test CSV format (`Handscribe/dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv`) or one of the Fieldwork languages' test CSV files (`Handscribe/gloss_generator/fieldwork_data/seen/selk1253.csv`).
- `NO_SENTS_TO_TEST`: the number of sentences to generate glosses for, taken from the end of the dataset (set to `0` to test all sentences).
- `MAX_SEQ_LENGTH`: the maximum sequence length the LLM will generate (e.g., `2048`); this should match the value used during fine-tuning.
- `DATA_LANGUAGE`: currently ignored as the language is auto-detected from `TEST_DATA_CSV_PATH` (German if PHOENIX-2014-T test CSV is used, and English otherwise).

> [!TIP]
> ❓ If you want to see the generated tokens as they are produced, add `--use_streamer`.

Example: running inference with the `unsloth/Llama-3.2-3B-Instruct` model fine-tuned above:

```
python Handscribe/recognizer/tuned_llms_inference.py --llm_to_use gloss_llama32_1k --data_csv_path Handscribe/dataset/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv --sentences_to_test 0 --max_seq_length 2048 --data_language German
```
