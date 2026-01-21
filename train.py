import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from models.language_model import ViVQAConfig, ViVQAForCausalLM

from data.dataset import VQADataset
from data.collator import VQACollator

from training.trainer import Trainer
from training.early_stopping import EarlyStopping
from training.evaluator import Evaluator
from utils.utils import countTrainableParameters, countAllParameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train decoder-only VQA with image-text fusion"
    )

    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--image_encoder_name", type=str, required=True, help="SigLIP vision tower name/path")
    parser.add_argument("--vision_projector_type", type=str, default="mlp2x_gelu",
                        help="Type of multimodal projector (e.g., mlp2x_gelu, linear, sppv1)")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None,
                        help="Optional path to pretrained mm_projector weights")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_best_path", type=str, default="outputs/best_model.pth")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_dev_samples", type=int, default=-1)

    parser.add_argument("--log_path", type=str, default="outputs/logs.json")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special token marking image position in the prompt
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

    # Initialize ViVQA model (Qwen3 + SigLIP vision tower)
    model = ViVQAForCausalLM.from_pretrained(
        args.llm_name,
        config=config,
        trust_remote_code=True,
    )

    # Initialize vision modules (SigLIP encoder + projector)
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

    # Resize token embeddings to account for added <image> token
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    print(f"Model has {countTrainableParameters(model):,} trainable parameters.")
    print(f"Model has {countAllParameters(model):,} total parameters.")

    train_set = VQADataset(
        data_path=args.train_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        vision_processor_name=args.image_encoder_name,
        max_sample=args.max_train_samples
    )
    dev_set = VQADataset(
        data_path=args.dev_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        vision_processor_name=args.image_encoder_name,
        max_sample=args.max_dev_samples
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=VQACollator(tokenizer)
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VQACollator(tokenizer)
    )

    # Only optimize trainable parameters (projector, image_norm)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        evaluator=evaluator,
        device=device,
        log_path=args.log_path
    )

    trainer.train(args.epochs, early_stopping=EarlyStopping(patience=args.patience), save_best_path=args.save_best_path)


if __name__ == "__main__":
    main()
