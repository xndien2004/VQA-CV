import torch
from torch.nn.utils.rnn import pad_sequence


class VQACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        bs = len(batch)

        # =======================
        # Vision + Text part
        # =======================
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

        det_feats_list = [b["ocr_det_features"] for b in batch]  # [Ni, D]
        rec_feats_list = [b["ocr_rec_features"] for b in batch]  # [Ni, D]
        boxes_list     = [b["ocr_boxes"] for b in batch]         # [Ni, 4]

        ocr_lengths = torch.tensor([x.size(0) for x in det_feats_list], dtype=torch.long)
        max_ocr_len = int(ocr_lengths.max().item())

        det_dim = det_feats_list[0].size(-1)
        rec_dim = rec_feats_list[0].size(-1)

        # pad features
        ocr_det_features = det_feats_list[0].new_zeros((bs, max_ocr_len, det_dim))
        ocr_rec_features = rec_feats_list[0].new_zeros((bs, max_ocr_len, rec_dim))
        ocr_boxes = boxes_list[0].new_zeros((bs, max_ocr_len, 4))

        for i in range(bs):
            n = det_feats_list[i].size(0)

            ocr_det_features[i, :n] = det_feats_list[i]
            ocr_rec_features[i, :n] = rec_feats_list[i]
            ocr_boxes[i, :n] = boxes_list[i]

        # height/width (scalar -> tensor)
        ocr_height = torch.tensor([b["ocr_height"] for b in batch], dtype=torch.long)
        ocr_width  = torch.tensor([b["ocr_width"] for b in batch], dtype=torch.long)

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "ocr_det_features": ocr_det_features,
            "ocr_rec_features": ocr_rec_features,
            "ocr_boxes": ocr_boxes,
            "ocr_height": ocr_height,
            "ocr_width": ocr_width,
        }
