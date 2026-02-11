import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from models.language_model.qwen import ViVQAConfig, ViVQAForCausalLM

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
    parser.add_argument("--caption_path", type=str, default=None)
    parser.add_argument("--ocr_path", type=str, default=None, help="Path to OCR features")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "linear", "cosine"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Portion of total steps used for LR warmup",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_dev_samples", type=int, default=-1)

    parser.add_argument("--sort_type", type=str, default="top-left bottom-right", help="OCR sorting type: random, score, top-left bottom-right, None")
    parser.add_argument("--scene_text_threshold", type=float, default=0.3, help="OCR score threshold to filter scene text")
    parser.add_argument("--d_det", type=int, default=256, help="Dimension of OCR detection features")
    parser.add_argument("--d_rec", type=int, default=256, help="Dimension of OCR recognition features")

    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to training log JSON (defaults to checkpoint_dir/logs.json)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save/load model, optimizer, and scheduler checkpoints",
    )

    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help="Epoch number to resume from using checkpoint_dir (model/optimizer/scheduler_epoch_*.pth)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.log_path is None:
        args.log_path = os.path.join(args.checkpoint_dir, "logs.json")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Add special tokens for image span markers.
    # These must match the tokens used in ViVQAProcessor.build_prompt
    # so that <image> positions are replaced by visual embeddings.
    special_tokens = {"additional_special_tokens": ["<image>", "<im_start>", "<im_end>"]}
    tokenizer.add_special_tokens(special_tokens)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_start_token_id = tokenizer.convert_tokens_to_ids("<im_start>")
    image_end_token_id = tokenizer.convert_tokens_to_ids("<im_end>")

    # Build ViVQA config from base Qwen3 config
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
        print("Configuring model to use OCR features from:", args.ocr_path)
        config.ocr_path = args.ocr_path
        config.sort_type = args.sort_type
        config.scene_text_threshold = args.scene_text_threshold
        config.d_det = args.d_det
        config.d_rec = args.d_rec
        config.max_scene_text = 32

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
        caption_path=args.caption_path,
        tokenizer=tokenizer,
        vision_processor_name=args.image_encoder_name,
        max_sample=args.max_train_samples
    )
    dev_set = VQADataset(
        data_path=args.dev_path,
        image_root=args.image_root,
        caption_path=args.caption_path,
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

    # Setup learning rate scheduler (optional)
    total_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * total_training_steps)

    if args.scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        scheduler = None

    # Load logs.json and resume from last finished epoch
    resume_epoch = None
    if os.path.exists(args.log_path):
        try:
            import json
            with open(args.log_path, "r") as f:
                logs = json.load(f)
            if logs:
                # Find last finished epoch
                resume_epoch = logs[-1]["epoch"] if "epoch" in logs[-1] else None
                print(f"Auto-detected resume epoch {resume_epoch} from logs.json")
        except Exception as e:
            print(f"Failed to load logs.json: {e}")

    if resume_epoch is not None:
        model_path = os.path.join(args.checkpoint_dir, f"model_epoch_{resume_epoch}.pth")
        optimizer_path = os.path.join(args.checkpoint_dir, f"optimizer_epoch_{resume_epoch}.pth")
        scheduler_path = os.path.join(args.checkpoint_dir, f"scheduler_epoch_{resume_epoch}.pth")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Resumed model weights from {model_path}")

        if os.path.exists(optimizer_path):
            opt_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(opt_state)
            print(f"Resumed optimizer state from {optimizer_path}")

        if scheduler is not None and os.path.exists(scheduler_path):
            sched_state = torch.load(scheduler_path, map_location=device)
            scheduler.load_state_dict(sched_state)
            print(f"Resumed scheduler state from {scheduler_path}")

    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        dev_loader=dev_loader,
        evaluator=evaluator,
        device=device,
        log_path=args.log_path,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train(args.epochs, early_stopping=EarlyStopping(patience=args.patience))


if __name__ == "__main__":
    main()
