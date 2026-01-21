import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from models.language_model import ViVQAConfig, ViVQAForCausalLM
from data.dataset import VQADataset

from training.evaluator import Evaluator
from data.collator import VQACollator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--image_encoder_name", type=str, required=True, help="SigLIP vision tower name/path")
    parser.add_argument("--vision_projector_type", type=str, default="mlp2x_gelu",
                        help="Type of multimodal projector (e.g., mlp2x_gelu, linear, sppv1)")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None,
                        help="Optional path to pretrained mm_projector weights")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output_path", type=str, default="predictions.json")

    return parser.parse_args()

@torch.no_grad()
def infer_and_eval(args):
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special image token consistent with training
    special_tokens = {"additional_special_tokens": ["<image>"]}
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # Build ViVQA config from base Qwen3 config
    base_config = AutoConfig.from_pretrained(args.llm_name, trust_remote_code=True)
    config_dict = base_config.to_dict()
    config = ViVQAConfig(**config_dict)
    config.mm_vision_tower = args.image_encoder_name
    config.mm_projector_type = args.vision_projector_type
    config.use_mm_proj = True
    config.image_token_id = image_token_id
    config.tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    config.tokenizer_padding_side = tokenizer.padding_side

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

    state_dict = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


    dataset = VQADataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        vision_processor_name=args.image_encoder_name,
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

    metrics, results = evaluator.evaluate(dataloader, return_predictions=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "results": results
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("===== EVALUATION RESULT =====")
    print(f"EM : {metrics['EM']}%")
    print(f"F1 : {metrics['F1']}%")
    print(f"Detailed results saved to {args.output_path}")

if __name__ == "__main__":
    args = parse_args()
    infer_and_eval(args)
