#!/bin/bash

# Set Python path
export PYTHONPATH="/home/fit02/dien_workspace/vqa/VQA-CV:$PYTHONPATH"

echo "Running training script..."

# Choose GPUs
export CUDA_VISIBLE_DEVICES=0

# Run training
python3 -m VQA-CV.train_sft \
    --dataset_path "/home/fit02/dien_workspace/vqa/dataset/viocrvqa/train.json" \
    --caption_path "/home/fit02/dien_workspace/vqa/dataset/viocrvqa/viocrvqa_captions.json" \
    --output_dir "/home/fit02/dien_workspace/vqa/outputs_sft_viocrvqa" \
    --model_name "Qwen/Qwen3-0.6B" \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --lr_scheduler_type linear \
    --logging_steps 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_input_length 2048 \
    --num_train_epochs 4 \
    2>&1 | tee Qwen-SFT.log