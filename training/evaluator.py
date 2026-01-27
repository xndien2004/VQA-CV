import torch
from tqdm import tqdm
from .metrics import exact_match, f1_score

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader, return_predictions=False):
        self.model.eval()

        total_samples = 0
        ems, f1s = [], []
        all_preds = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            bs = batch["labels"].size(0)
            total_samples += bs

            generated_ids = self.model.generate(
                input_ids=batch["prompt_ids"],
                images=batch["images"],
                max_new_tokens=64,
                do_sample=False
            )
            generated_ids = generated_ids[:, batch["prompt_ids"].size(1):]
            preds_text = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            labels_clean = batch["labels"].clone()
            labels_clean[labels_clean == -100] = self.tokenizer.pad_token_id
            labels_text = self.tokenizer.batch_decode(
                labels_clean, skip_special_tokens=True
            )

            for p, g in zip(preds_text, labels_text):
                em = exact_match(p, g)
                f1 = f1_score(p, g)
                print(f"Pred: {p} | Gold: {g} | EM: {em} | F1: {f1}")
                ems.append(em)
                f1s.append(f1)

            all_preds.extend(preds_text)

        report = {
            "EM": sum(ems) / len(ems),
            "F1": sum(f1s) / len(f1s)
        }

        if return_predictions:
            return report, all_preds, ems, f1s
        return report


