#!/bin/bash

# ===== CONFIG =====
# GPU_ID=1
# MEM_THRESHOLD=500      # MB
# UTIL_THRESHOLD=5       # %
# CHECK_INTERVAL=60      # seconds
# LOG_DIR="/home/fit02/dien_workspace/vqa/logs"
# mkdir -p $LOG_DIR

# echo "Waiting for GPU $GPU_ID to be free..."

# while true; do
#     MEM_USED=$(nvidia-smi \
#         --id=$GPU_ID \
#         --query-gpu=memory.used \
#         --format=csv,noheader,nounits)

#     UTIL=$(nvidia-smi \
#         --id=$GPU_ID \
#         --query-gpu=utilization.gpu \
#         --format=csv,noheader,nounits)

#     PROC=$(nvidia-smi \
#         --id=$GPU_ID \
#         --query-compute-apps=pid \
#         --format=csv,noheader)

#     if [[ "$MEM_USED" -lt "$MEM_THRESHOLD" ]] && \
#        [[ "$UTIL" -lt "$UTIL_THRESHOLD" ]] && \
#        [[ -z "$PROC" ]]; then
#         echo "GPU $GPU_ID is free. Starting evaluation..."
#         break
#     fi

#     echo "GPU busy (mem=${MEM_USED}MB, util=${UTIL}%). Recheck in ${CHECK_INTERVAL}s..."
#     sleep $CHECK_INTERVAL
# done

export PYTHONPATH="/home/fit02/dien_workspace/vqa/VQA-CV:$PYTHONPATH"
echo "Running evaluation script..."

export CUDA_VISIBLE_DEVICES=1

DATA_PATH="/home/fit02/dien_workspace/vqa/dataset/ViTextVQA_test_gt.json"
IMAGE_ROOT="/home/fit02/dien_workspace/vqa/dataset/images/st_images"
OUTPUT_PATH="/home/fit02/dien_workspace/vqa/outputs"

python3 -m VQA-CV.infer_eval \
  --llm_name "Qwen/Qwen3-0.6B" \
  --image_encoder_name "google/siglip2-so400m-patch16-naflex" \
  --vision_projector_type "mlp2x_gelu" \
  --data_path "${DATA_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --batch_size 50 \
  --max_new_tokens 64 \
  --device "cuda" \
  --output_path "${OUTPUT_PATH}" \
  2>&1 | tee eval.log
