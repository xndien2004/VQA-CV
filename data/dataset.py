import torch
import json
import ast
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import SiglipImageProcessor

from models.processor import ViVQAProcessor, STRICT_SYSTEM_PROMPT

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

        self.data = self._load_data()
        if max_sample > 0:
            self.data = self.data[:max_sample]

    def _load_data(self):
        if self._is_json():
            return self._load_json()
        return self._load_csv()

    def _is_json(self):
        path = str(self.data_path).lower()
        return "vitextvqa" in path or "viocrvqa" in path

    def _load_json(self):
        with open(self.data_path, encoding="utf-8") as f:
            raw = json.load(f)

        df_img = pd.DataFrame(raw["images"])
        df_ann = pd.DataFrame(raw["annotations"])

        df = df_ann.merge(df_img, left_on="image_id", right_on="id")

        df["answer"] = df["answers"].apply(
            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x[0]
        )

        return df.to_dict("records")

    def _load_csv(self):
        return pd.read_csv(self.data_path).to_dict("records")

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
