import argparse
import json
import torch
import os
import random
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from models.language_model.qwen import ViVQAConfig, ViVQAForCausalLM
from data.dataset import VQADataset
from training.evaluator import Evaluator
from data.collator import VQACollator
from utils.plot import plot_image_predictions

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--image_encoder_name", type=str, required=True, help="SigLIP vision tower name/path")
    parser.add_argument("--vision_projector_type", type=str, default="mlp2x_gelu",
                        help="Type of multimodal projector (e.g., mlp2x_gelu, linear, sppv1)")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None,
                        help="Optional path to pretrained mm_projector weights")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--caption_path", type=str, default=None)

    parser.add_argument("--ocr_path", type=str, default=None, help="Path to OCR features")
    parser.add_argument("--sort_type", type=str, default="top-left bottom-right",
                        help="OCR sorting type: random, score, top-left bottom-right, None")
    parser.add_argument("--scene_text_threshold", type=float, default=0.3,
                        help="OCR score threshold to filter scene text")
    parser.add_argument("--d_det", type=int, default=256,
                        help="Dimension of OCR detection features")
    parser.add_argument("--d_rec", type=int, default=256,
                        help="Dimension of OCR recognition features")
    parser.add_argument("--max_scene_text", type=int, default=32,
                        help="Maximum number of OCR tokens to consider")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length for tokenizer")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_path", type=str, default="outputs")

    return parser.parse_args()

@torch.no_grad()
def infer_and_eval(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    special_tokens = {"additional_special_tokens": ["<image>", "<im_start>", "<im_end>"]}
    tokenizer.add_special_tokens(special_tokens)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_start_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    image_end_token_id = tokenizer.convert_tokens_to_ids("<im_end>")

    base_config = AutoConfig.from_pretrained(args.llm_name, trust_remote_code=True)
    config_dict = base_config.to_dict()
    config = ViVQAConfig(**config_dict)
    config.mm_vision_tower = args.image_encoder_name
    config.mm_projector_type = args.vision_projector_type
    config.use_mm_proj = True
    config.image_token_id = image_token_id
    config.image_start_token_id = image_start_token_id
    config.image_end_token_id = image_end_token_id
    config.tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    config.tokenizer_padding_side = tokenizer.padding_side

    if args.ocr_path is not None:
        print(f"Configuring model to use OCR features from: {args.ocr_path}")
        config.ocr_path = args.ocr_path
        config.sort_type = args.sort_type
        config.scene_text_threshold = args.scene_text_threshold
        config.d_det = args.d_det
        config.d_rec = args.d_rec
        config.max_scene_text = args.max_scene_text

    model = ViVQAForCausalLM.from_pretrained(
        args.llm_name,
        config=config,
        trust_remote_code=True,
    )

    # Initialize vision modules
    class ModelArgs:
        def __init__(self, vision_tower, mm_projector_type, pretrain_mm_mlp_adapter=None):
            self.vision_tower = vision_tower
            self.mm_projector_type = mm_projector_type
            self.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter

    model_args = ModelArgs(
        vision_tower=args.image_encoder_name,
        mm_projector_type=args.vision_projector_type,
        pretrain_mm_mlp_adapter=args.pretrain_mm_mlp_adapter,
    )
    model.get_model().initialize_vision_modules(model_args)

    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(f"{args.output_path}/best_model.pth")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()


    dataset = VQADataset(
        data_path=args.data_path,
        image_root=args.image_root,
        caption_path=args.caption_path,
        tokenizer=tokenizer,
        vision_processor_name=args.image_encoder_name,
        max_length=args.max_length
        # max_sample=20
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=VQACollator(tokenizer),
        num_workers=2
    )

    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    metrics, results, ems, f1s = evaluator.evaluate(dataloader, return_predictions=True)

    data_origin = dataset.data
    data_new = []
    for data_final, result, em, f1 in zip(data_origin, results, ems, f1s):
        data_final["predicted_answer"] = result

        data_final["EM"] = em
        data_final["F1"] = f1

        image_path = dataset.image_root / data_final["filename"]

        data_new.append({
            "image_path": str(image_path),
            "question": data_final["question"],
            "ground_truth": data_final["answer"],
            "prediction": data_final["predicted_answer"],
            "EM": data_final["EM"],
            "F1": data_final["F1"],
            "EM_all": metrics['EM'],
            "F1_all": metrics['F1']
        })

    
    with open(f"{args.output_path}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(data_new, f, ensure_ascii=False, indent=4)

    data0 = [data for data in data_new if data["EM"] == 0]
    data1 = [data for data in data_new if data["EM"] == 1]

    for item in data0:
        item["image"] = Image.open(item["image_path"]).convert("RGB")

    for item in data1:
        item["image"] = Image.open(item["image_path"]).convert("RGB")
    try:
        plot_image_predictions(data0[:6], f"{args.output_path}/incorrect_predictions.png")
        plot_image_predictions(data1[:6], f"{args.output_path}/correct_predictions.png")
    except Exception as e:
        print(f"Error in plotting image predictions: {e}")


    print("===== EVALUATION RESULT =====")
    print(f"EM : {metrics['EM']}%")
    print(f"F1 : {metrics['F1']}%")
    print(f"Detailed results saved to {args.output_path}")

if __name__ == "__main__":
    args = parse_args()
    infer_and_eval(args)
