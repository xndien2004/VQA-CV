from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import json
from math import ceil

from .utils.utils import load_data
from .training.metrics import exact_match, f1_score

SYSTEM_PROMPT = "Bạn là trợ lý chuyên trả lời câu hỏi dựa trên thông tin được cung cấp. Hãy trả lời một cách ngắn gọn và chính xác."


def build_prompt(question, caption):
    user_prompt = f"Câu hỏi: {question}\nMô tả hình ảnh: {caption}\nTrả lời ngắn gọn (chỉ đáp án):"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    ).to(device)

    # Load dataset
    with open(args.caption_path, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)

    dataset = load_data(args.dataset_path)

    results = []
    batch_size = args.batch_size

    total_batches = ceil(len(dataset) / batch_size)

    for batch_idx in range(total_batches):

        batch_df = dataset.iloc[
            batch_idx * batch_size:(batch_idx + 1) * batch_size
        ]

        prompts = []
        questions = []
        answers = []
        filenames = []

        for row in batch_df.itertuples():
            caption = caption_data.get(row.filename, "")
            messages = build_prompt(row.question, caption)

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            prompts.append(text)
            questions.append(row.question)
            answers.append(row.answer)
            filenames.append(row.filename)

        # Tokenize batch
        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=64
            )

        # Extract outputs từng sample
        for i in range(len(prompts)):
            input_len = model_inputs.input_ids[i].shape[0]
            output_ids = generated_ids[i][input_len:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            res = {
                "image_path": filenames[i],
                "question": questions[i],
                "ground_truth": answers[i],
                "prediction": output_text,
                "EM": exact_match(output_text, answers[i]),
                "F1": f1_score(output_text, answers[i])
            }

            results.append(res)

            print(
                f"Q: {questions[i]}\n"
                f"GT: {answers[i]}\n"
                f"Pred: {output_text}\n"
                f"EM: {res['EM']}, F1: {res['F1']}\n---"
            )

    # ===== Average metrics =====
    avg_em = sum(r["EM"] for r in results) / len(results)
    avg_f1 = sum(r["F1"] for r in results) / len(results)

    final_output = {
        "results": results,
        "Average_EM": avg_em,
        "Average_F1": avg_f1
    }

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "eval_results.json"), 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--caption_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    main(args)
