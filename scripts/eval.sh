export PYTHONPATH="/home/fit02/dien_workspace/vqa/VQA-CV:$PYTHONPATH"
echo "Running evaluation script..."

export CUDA_VISIBLE_DEVICES=1

CKPT="outputs/best_model.pth"

DATA_PATH="/home/fit02/dien_workspace/vqa/dataset/ViTextVQA_test_gt.json"
IMAGE_ROOT="/home/fit02/dien_workspace/vqa/dataset/images/st_images"
OUTPUT_PATH="outputs/predictions_test.json"

python3 -m VQA-CV.infer_eval \
  --ckpt "${CKPT}" \
  --llm_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --image_encoder_name "google/siglip2-so400m-patch14-384" \
  --vision_projector_type "mlp2x_gelu" \
  --data_path "${DATA_PATH}" \
  --image_root "${IMAGE_ROOT}" \
  --batch_size 4 \
  --max_new_tokens 64 \
  --device "cuda" \
  --output_path "${OUTPUT_PATH}" \
  2>&1 | tee eval.log
