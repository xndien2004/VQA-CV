# https://huggingface.co/datasets/nhonhoccode/RecieptVQA
# https://huggingface.co/datasets/nhonhoccode/ViOCRVQA
# https://huggingface.co/datasets/nhonhoccode/ViTextVQA

OUTPUT_DIR="./datasets"

# Usage:
#   bash scripts/download_dataset.sh
#   bash scripts/download_dataset.sh --dataset vitextvqa

python data_preparation/download_dataset.py \
    --output_dir "$OUTPUT_DIR" \
    "$@"
