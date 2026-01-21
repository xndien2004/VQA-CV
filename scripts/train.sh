
export PYTHONPATH="/kaggle/working/VQA-CV:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=0

python3 -m VQA-CV.train 	--llm_name Qwen/Qwen2.5-0.5B-Instruct     --image_encoder_name google/siglip-so400m-patch14-384     --vision_projector_type mlp2x_gelu     --train_path /kaggle/input/datavqa/data_VQA/RecieptVQA/train.csv     --dev_path /kaggle/input/datavqa/data_VQA/RecieptVQA/dev.csv     --image_root /kaggle/input/datavqa/data_VQA/RecieptVQA/images     --epochs 30     --batch_size 1     --lr 2e-4     --patience 5     --save_best_path outputs/best_model.pth     --max_train_samples 2000     --max_dev_samples 40     2>&1 | tee train.log
    