import torch
from typing import Any, Dict, Optional

from transformers import AutoTokenizer, SiglipImageProcessor

from data.preprocessing import preprocess_sentence


STRICT_SYSTEM_PROMPT = "Bạn là trợ lý chuyên trả lời câu hỏi dựa trên nội dung văn bản xuất hiện trong hình ảnh và các thông tin khác bổ sung từ hình ảnh (nếu có)."

class ViVQAProcessor:
    """Unified image + text processor for ViVQA (Qwen2 + SigLIP).

    - Manages the LLM tokenizer and the SigLIP image processor.
    - Defines a fixed VQA prompt template.
    - Returns fields compatible with the current dataloader/trainer.
    """

    def __init__(
        self,
        tokenizer,
        image_processor: SiglipImageProcessor,
        system_prompt: str = STRICT_SYSTEM_PROMPT,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.system_prompt = system_prompt

    @classmethod
    def from_pretrained(
        cls,
        llm_name: str,
        vision_tower_name: str,
        system_prompt: str = STRICT_SYSTEM_PROMPT,
        **image_kwargs: Any,
    ) -> "ViVQAProcessor":
        """Initialize the processor directly from LLM and vision tower names.

        Example:
            processor = ViVQAProcessor.from_pretrained(
                llm_name="Qwen/Qwen2-7B-Instruct",
                vision_tower_name="google/siglip-so400m-patch14-384",
            )
        """
        tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Add special tokens for image and image span markers
        # Keep the order consistent with train.py / infer_eval.py
        special_tokens = {"additional_special_tokens": ["<image>", "<im_start>", "<im_end>"]}
        tokenizer.add_special_tokens(special_tokens)

        image_processor = SiglipImageProcessor.from_pretrained(vision_tower_name, **image_kwargs)
        image_processor.crop_size = image_processor.size

        return cls(tokenizer=tokenizer, image_processor=image_processor, system_prompt=system_prompt)

    def build_prompt(self, question: str, caption: Optional[str] = None) -> str:
        caption_text = f"Mô tả hình ảnh: {caption}\n" if caption is not None else ""

        return (
            "<|im_start|>system\n"
            f"{self.system_prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<im_start> <image> <im_end>\n"
            f"Câu hỏi: {question}\n"
            f"{caption_text}"
            "Trả lời ngắn gọn (chỉ đáp án):"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def preprocess_train(
        self,
        image,
        question: str,
        answer: str,
        caption: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single training example (image + question + answer).

        Returns the same format that the original VQADataset produced.
        """
        question = preprocess_sentence(question)
        answer = preprocess_sentence(answer)

        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)
        prompt = self.build_prompt(question, caption=caption)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, max_length=2048, truncation=True)
        answer_ids = self.tokenizer.encode(answer + " <|im_end|>", add_special_tokens=False)

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        answer_ids = torch.tensor(answer_ids, dtype=torch.long)

        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100  # ignore prompt tokens when computing loss

        return {
            "images": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "prompt_ids": prompt_ids
        }


    def preprocess_infer(
        self,
        image,
        question: str,
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a single inference example (image + question).

        - Returns pixel_values and prompt_ids ready for model.generate.
        - add_generation_prompt keeps the final "assistant" block like in training.
        """
        question = preprocess_sentence(question)

        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        prompt = self.build_prompt(question)
        if not add_generation_prompt:
            prompt = prompt.replace("<|im_start|>assistant\n", "")

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

        return {
            "images": pixel_values,
            "prompt_ids": prompt_ids
        }
