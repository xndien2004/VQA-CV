export PYTHONPATH="/home/fit02/dien_workspace/vqa/VQA-CV:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=1

python3 -m VQA-CV.train \
	--llm_name Qwen/Qwen2.5-0.5B-Instruct \
    --image_encoder_name google/siglip-so400m-patch14-384 \
    --vision_projector_type mlp2x_gelu \
    --train_path /home/fit02/dien_workspace/vqa/dataset/ViTextVQA_train.json \
    --dev_path /home/fit02/dien_workspace/vqa/dataset/ViTextVQA_dev.json \
    --image_root /home/fit02/dien_workspace/vqa/dataset/images/st_images \
    --epochs 20 \
    --batch_size 6 \
    --lr 2e-4 \
    --patience 3 \
    --checkpoint_dir outputs/ \
    --max_train_samples 100 \
    --max_dev_samples 10 \
    2>&1 | tee train0.5B.log