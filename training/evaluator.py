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

        total_loss = 0
        total_samples = 0
        ems, f1s = [], []
        all_preds = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["images"],
                labels=batch["labels"],
            )

            bs = batch["labels"].size(0)
            total_loss += outputs.loss.item() * bs
            total_samples += bs

            prompt_attention_mask = (batch["prompt_ids"] != self.tokenizer.pad_token_id).long()
            prompt_attention_mask = prompt_attention_mask.to(self.device)

            generated_ids = self.model.generate(
                input_ids=batch["prompt_ids"],
                attention_mask=prompt_attention_mask,
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
                print(f"Pred: {p} | Gold: {g}")
                ems.append(exact_match(p, g))
                f1s.append(f1_score(p, g))

            all_preds.extend(preds_text)

        report = {
            "loss": total_loss / total_samples,
            "EM": sum(ems) / len(ems),
            "F1": sum(f1s) / len(f1s)
        }

        if return_predictions:
            return report, all_preds
        return report


