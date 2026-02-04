import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import os
import json

from .utils.utils import load_data

SYSTEM_PROMPT = "Bạn là trợ lý chuyên trả lời câu hỏi dựa trên thông tin được cung cấp. Hãy trả lời một cách ngắn gọn và chính xác."

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    with open(args.caption_path, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)

    train_dataset = load_data(args.dataset_path)
    data_list = []
    for i in train_dataset.itertuples():
        question = i.question
        answer = i.answer
        caption = caption_data.get(i.filename, "")
        user_prompt = "Câu hỏi: {}\nMô tả hình ảnh: {}\nTrả lời ngắn gọn (chỉ đáp án):".format(
            question, caption
        )
        data_list.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "completion": [
                {"role": "assistant", "content": answer}
            ]
        })
    train_dataset = Dataset.from_list(data_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    ).to(device)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=2,
        report_to="none",
        completion_only_loss=False,
        deepspeed=args.deepspeed_path,
        max_length=args.max_input_length,
        # eval_strategy="steps",
        # eval_steps=args.eval_steps
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        args=training_args
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--caption_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_lora_output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--deepspeed_path", type=str, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_input_length", type=int, default=1024)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)