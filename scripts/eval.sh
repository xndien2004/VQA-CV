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

export CUDA_VISIBLE_DEVICES=0

python3 -m VQA-CV.infer_eval \
  --llm_name Qwen/Qwen3-0.6B \
  --image_encoder_name google/siglip2-so400m-patch16-naflex \
  --vision_projector_type mlp2x_gelu \
  --data_path /home/fit02/dien_workspace/vqa/dataset/recieptvqa/test.csv \
  --image_root /home/fit02/dien_workspace/vqa/dataset/recieptvqa/images \
  --caption_path /home/fit02/dien_workspace/vqa/dataset/recieptvqa/recieptvqa_captions.json \
  --ocr_path /home/fit02/dien_workspace/vqa/dataset/recieptvqa/ocr_features_for_recipevqa.npy \
  --batch_size 2 \
  --max_new_tokens 64 \
  --device cuda \
  --output_path /home/fit02/dien_workspace/vqa/outputs_ocr \
  2>&1 | tee eval.log
