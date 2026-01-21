# VQA-CV: Visual Question Answering with Qwen2 + SigLIP

This repository contains a simple Visual Question Answering (VQA) system for receipts and similar documents. The model answers short questions about an image (e.g., brand name, price) with a single, concise text answer.

## Architecture Overview

- **Language model**: Qwen2 / Qwen2.5 (decoder-only, causal LM).
- **Vision encoder**: SigLIP `google/siglip-so400m-patch14-384`.
- **Multimodal fusion**:
  - Images are encoded into patch features by SigLIP.
  - A projector maps visual features into the Qwen hidden space.
  - Visual tokens are inserted at `<image>` positions inside the text sequence.
  - Training and inference are both done via a single causal LM head (next-token prediction).

## Run on Kaggle with GitHub token

In a Kaggle Notebook, you can clone the repo with a personal access token and install dependencies as follows (replace `<TOKEN>` with your GitHub token):

```bash
!git clone https://<TOKEN>@github.com/xndien2004/VQA-CV.git
!pip install -r /kaggle/working/VQA-CV/requirements.txt
``

## Training with bash (train.sh)

After cloning and installing requirements, you can launch training using the provided shell script. In a Kaggle Notebook (or any bash environment):

```bash
%cd /kaggle/working/VQA-CV
!bash scripts/train.sh
```

The default `scripts/train.sh` script:

- Sets `PYTHONPATH` to point to the cloned repo.
- Calls `python3 -m VQA-CV.train` with reasonable defaults:
  - `--llm_name Qwen/Qwen2.5-0.5B-Instruct`
  - `--image_encoder_name google/siglip-so400m-patch14-384`
  - `--train_path`, `--dev_path`, `--image_root` pointing to the Kaggle dataset paths.

To adapt training to your own data, edit `scripts/train.sh` and change at least:

- `--train_path` and `--dev_path` to your CSV/JSON files.
- `--image_root` to the folder that contains your images.
- Optionally `--epochs`, `--batch_size`, `--lr`, `--max_train_samples`, `--max_dev_samples`.

Once these paths and hyperparameters are set, rerun:

```bash
!bash scripts/train.sh
```

This will start training the multimodal VQA model and save the best checkpoint to `outputs/best_model.pth`.
