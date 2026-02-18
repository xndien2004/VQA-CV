import torch
from torch.nn.utils.rnn import pad_sequence


class VQACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        bs = len(batch)
        images = torch.stack([b["images"] for b in batch])

        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        labels = pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=-100
        )

        prompt_seqs = [b["prompt_ids"] for b in batch]
        max_prompt_len = max(seq.size(0) for seq in prompt_seqs)

        prompt_ids = input_ids.new_full(
            (bs, max_prompt_len),
            fill_value=self.tokenizer.pad_token_id
        )

        for i, seq in enumerate(prompt_seqs):
            prompt_ids[i, -seq.size(0):] = seq

        det_feats_list = [b["ocr_det_features"] for b in batch] if "ocr_det_features" in batch[0] else None  # [Ni, D]
        rec_feats_list = [b["ocr_rec_features"] for b in batch] if "ocr_rec_features" in batch[0] else None  # [Ni, D]
        boxes_list     = [b["ocr_boxes"] for b in batch] if "ocr_boxes" in batch[0] else None  # [Ni, 4]
        ocr_height = torch.tensor([b["ocr_height"] for b in batch], dtype=torch.long) if "ocr_height" in batch[0] else None
        ocr_width  = torch.tensor([b["ocr_width"] for b in batch], dtype=torch.long) if "ocr_width" in batch[0] else None

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "ocr_det_features": det_feats_list,
            "ocr_rec_features": rec_feats_list,
            "ocr_boxes": boxes_list,
            "ocr_height": ocr_height,
            "ocr_width": ocr_width,
        }
