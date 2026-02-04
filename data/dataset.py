import torch
import json
import ast
import pandas as pd
import numpy as np
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
        caption_path: str = None,
        tokenizer: any = None,
        vision_processor_name: str=None,
        max_sample: int = -1,
    ):
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)
        self.caption_path = caption_path
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
    
    def _get_caption(self, image_filename):
        if self.caption_path is None:
            return None
        with open(self.caption_path, 'r', encoding="utf-8") as f:
            captions = json.load(f)
        return captions.get(image_filename, None)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self._load_image(item["filename"])
        sample = self.processor.preprocess_train(
            image=image,
            question=item["question"],
            answer=item["answer"],
            caption=self._get_caption(item["filename"])
        )

        return {
            "images": sample["images"],
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
            "prompt_ids": sample["prompt_ids"],
            "image_id": int(item["filename"].replace(".jpg", "").replace(".jpeg", "").replace(".png", "")),
        }

    def _load_image(self, filename):
        return Image.open(self.image_root / filename).convert("RGB")
