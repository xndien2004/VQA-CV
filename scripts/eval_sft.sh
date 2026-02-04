
export PYTHONPATH="/home/fit02/dien_workspace/vqa/VQA-CV:$PYTHONPATH"
echo "Running evaluation script..."

export CUDA_VISIBLE_DEVICES=0

python3 -m VQA-CV.eval_sft \
  --model_name /home/fit02/dien_workspace/vqa/outputs_sft_vitextvqa \
    --dataset_path "/home/fit02/dien_workspace/vqa/dataset/vitextvqa/ViTextVQA_test_gt.json" \
    --caption_path "/home/fit02/dien_workspace/vqa/dataset/vitextvqa/vitextvqa_captions.json" \
    --output_dir "/home/fit02/dien_workspace/vqa/outputs_sft_vitextvqa" \
    --batch_size 64 \
  2>&1 | tee eval_sft.log
