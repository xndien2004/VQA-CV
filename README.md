# VQA-CV: Visual Question Answering for Vietnamese Documents

This repository contains a multimodal Visual Question Answering (VQA) system for Vietnamese document images (receipts, scene text, OCR-based QA). The model answers questions about an image with a concise text answer.

## Architecture Overview

- **Language model**: Qwen2.5 / Qwen3 (decoder-only, causal LM).
- **Vision encoder**: SigLIP2 `google/siglip2-so400m-patch16-naflex` (or C-RADIO / DINOv2).
- **OCR encoder**: encodes scene-text features from pre-extracted OCR embeddings (`.npy`).
- **Multimodal fusion**:
  - Images are encoded into patch features by the vision encoder.
  - A projector maps visual features into the LLM hidden space.
  - OCR token features are fused alongside visual tokens.
  - Training and inference use a single causal LM head (next-token prediction).

## Datasets

| Key | HuggingFace repo |
|---|---|
| `recieptvqa` | [nhonhoccode/RecieptVQA](https://huggingface.co/datasets/nhonhoccode/RecieptVQA) |
| `viocrvqa` | [nhonhoccode/ViOCRVQA](https://huggingface.co/datasets/nhonhoccode/ViOCRVQA) |
| `vitextvqa` | [nhonhoccode/ViTextVQA](https://huggingface.co/datasets/nhonhoccode/ViTextVQA) |

## Download Datasets

Set `OUTPUT_DIR` in `scripts/download_dataset.sh`, then run:

```bash
# Download all datasets
bash scripts/download_dataset.sh

# Download a single dataset
bash scripts/download_dataset.sh --dataset vitextvqa
```

Datasets will be saved to `OUTPUT_DIR/<key>/` (e.g. `./datasets/vitextvqa/`). Any `images.zip` found inside will be extracted automatically.

To download via Python directly:

```bash
python data_preparation/download_dataset.py --output_dir ./datasets --dataset all
```

## Run on Kaggle with GitHub token

In a Kaggle Notebook, clone the repo with a personal access token and install dependencies (replace `<TOKEN>` with your GitHub token):

```bash
!git clone https://<TOKEN>@github.com/xndien2004/VQA-CV.git
!pip install -r /kaggle/working/VQA-CV/requirements.txt
```

## Training

Edit `scripts/train.sh` to set at least:

- `--train_path` and `--dev_path` — JSON annotation files.
- `--image_root` — folder containing images.
- `--ocr_path` — `.npy` file with pre-extracted OCR features (optional).

Then run:

```bash
bash scripts/train.sh
```

Key training arguments:

| Argument | Description |
|---|---|
| `--llm_name` | HuggingFace LLM (e.g. `Qwen/Qwen3-0.6B`) |
| `--image_encoder_name` | Vision encoder (e.g. `google/siglip2-so400m-patch16-naflex`) |
| `--epochs` | Number of training epochs |
| `--batch_size_train` | Training batch size |
| `--lr` | Learning rate |
| `--patience` | Early stopping patience |
| `--checkpoint_dir` | Directory to save best checkpoint |

## Evaluation

```bash
bash scripts/eval.sh
```

## Project Structure

```
VQA-CV/
├── data/                   # Dataset & collator classes
├── data_preparation/       # Download & preprocessing scripts
│   └── download_dataset.py
├── models/                 # Model architecture
│   ├── language_model/
│   ├── multimodal_encoder/
│   └── ocr_encoder/
├── scripts/                # Shell scripts for training / eval / download
│   ├── download_dataset.sh
│   ├── train.sh
│   ├── train_sft.sh
│   └── eval.sh
├── training/               # Trainer, evaluator, metrics
├── utils/
├── train.py
├── train_sft.py
└── requirements.txt
```
