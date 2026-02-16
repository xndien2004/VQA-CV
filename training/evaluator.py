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

        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            ocr_keys = [k for k in batch.keys() if k.startswith("ocr_")]
            ocr_batch = {k: batch[k] for k in ocr_keys}
            non_ocr_batch = {k: v for k, v in batch.items() if not k.startswith("ocr_")}

            moved_batch = {}
            for k, v in non_ocr_batch.items():
                try:
                    moved_batch[k] = v.to(self.device)
                except Exception:
                    moved_batch[k] = v
            for k, v in ocr_batch.items():
                moved_batch[k] = v
            batch = moved_batch

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.autocast_dtype):
                ocr_keys = [k for k in batch.keys() if k.startswith("ocr_")]
                if len(ocr_keys) > 0:
                    batch_size = batch["images"].size(0)
                    ocr_info_list = []
                    for i in range(batch_size):
                        sample_ocr = {}
                        for k in ocr_keys:
                            v = batch[k]
                            try:
                                if isinstance(v, torch.Tensor):
                                    sample_ocr[k] = v[i]
                                else:
                                    # assume list-like
                                    sample_ocr[k] = v[i]
                            except Exception:
                                sample_ocr[k] = None
                        ocr_info_list.append(sample_ocr)
                else:
                    ocr_info_list = None
            bs = batch["labels"].size(0)
            total_samples += bs

            gen_kwargs = dict(
                input_ids=batch["prompt_ids"],
                images=batch["images"],
                ocr_info=ocr_info_list,
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
                # print(f"Pred: {p} | Gold: {g} | EM: {em} | F1: {f1}")
                ems.append(em)
                f1s.append(f1)

            all_preds.extend(preds_text)

            # free large temporaries per batch
            try:
                del generated_ids
            except Exception:
                pass
            try:
                del labels_clean
            except Exception:
                pass
            try:
                del ocr_info_list
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Update progress bar postfix with running F1/EM
            if len(ems) > 0:
                pbar.set_postfix(F1=f"{sum(f1s)/len(f1s):.4f}", EM=f"{sum(ems)/len(ems):.4f}")

        report = {
            "EM": sum(ems) / len(ems),
            "F1": sum(f1s) / len(f1s)
        }

        if return_predictions:
            return report, all_preds, ems, f1s
        return report


