import torch
import json
import ast
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import SiglipImageProcessor

from models.processor import ViVQAProcessor, STRICT_SYSTEM_PROMPT
from utils.utils import load_data

class VQADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_root: str,
        tokenizer,
        vision_processor_name: str,
        max_sample: int = -1,
    ):
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer

        image_processor = SiglipImageProcessor.from_pretrained(vision_processor_name)
        image_processor.crop_size = image_processor.size
        self.processor = ViVQAProcessor(tokenizer=self.tokenizer, image_processor=image_processor,
                        system_prompt=STRICT_SYSTEM_PROMPT)

        self.data = load_data(self.data_path).to_dict("records")
        if max_sample > 0:
            self.data = self.data[:max_sample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self._load_image(item["filename"])
        sample = self.processor.preprocess_train(
            image=image,
            question=item["question"],
            answer=item["answer"],
        )

        return {
            "images": sample["images"],
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
            "prompt_ids": sample["prompt_ids"],
        }

    def _load_image(self, filename):
        return Image.open(self.image_root / filename).convert("RGB")
