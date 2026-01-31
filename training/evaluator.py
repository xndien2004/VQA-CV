import torch
from tqdm import tqdm
from .metrics import exact_match, f1_score

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = self.device.type == "cuda"
        if self.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float32

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

            gen_kwargs = dict(
                input_ids=batch["prompt_ids"],
                images=batch["images"],
                image_ids=batch.get("image_ids", None),
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
            )

            if return_predictions:
                gen_kwargs["num_beams"] = 3

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.autocast_dtype):
                generated_ids = self.model.generate(**gen_kwargs)

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


