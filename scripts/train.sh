export PYTHONPATH="/home/fit02/dien_workspace/vqa/VQA-CV:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=1

# Qwen/Qwen2.5-0.5B-Instruct 
# Qwen/Qwen3-0.6B

# google/siglip2-so400m-patch16-512
# google/siglip2-so400m-patch16-naflex
python3 -m VQA-CV.train \
	--llm_name Qwen/Qwen3-0.6B \
    --image_encoder_name google/siglip2-so400m-patch16-naflex \
    --vision_projector_type mlp2x_gelu \
    --train_path /home/fit02/dien_workspace/vqa/dataset/viocrvqa/train.json \
    --dev_path /home/fit02/dien_workspace/vqa/dataset/viocrvqa/dev.json \
    --image_root /home/fit02/dien_workspace/vqa/dataset/viocrvqa/images \
    --caption_path /home/fit02/dien_workspace/vqa/dataset/viocrvqa/viocrvqa_captions.json \
    --ocr_path /home/fit02/dien_workspace/vqa/dataset/viocrvqa/docr_features_of_viocr.npy \
    --epochs 50 \
    --batch_size 4 \
    --lr 2e-5 \
    --patience 3 \
    --checkpoint_dir outputs_ocr_vqa/ \
    --max_train_samples -1 \
    --max_dev_samples -1 \
    2>&1 | tee train0.6B_ocr_vqa.log