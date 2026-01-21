import torch
from torch.nn.utils.rnn import pad_sequence


class VQACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
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
            (len(prompt_seqs), max_prompt_len),
            fill_value=self.tokenizer.pad_token_id,
        )
        for i, seq in enumerate(prompt_seqs):
            prompt_ids[i, -seq.size(0):] = seq

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_ids": prompt_ids,
        }
