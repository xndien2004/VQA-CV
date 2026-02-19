from typing import List, Optional, Tuple

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2Config, Qwen2ForCausalLM

from ..vivqa_arch import ViVQAMetaForCausalLM, ViVQAMetaModel


class ViVQAConfig(Qwen2Config):
    model_type = "vivqa"


class ViVQAModel(Qwen2Model, ViVQAMetaModel):
    config_class = ViVQAConfig

    def __init__(self, config: Qwen2Config):
        Qwen2Model.__init__(self, config)
        ViVQAMetaModel.__init__(self, config)


class ViVQAForCausalLM(Qwen2ForCausalLM, ViVQAMetaForCausalLM):

    config_class = ViVQAConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ViVQAModel(config)

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            ocr_info: Optional[dict] = None,
            **kwargs,
            ) -> Tuple:
        assert ocr_info is not None, "ocr_info none at ViVQAForCausalLM."
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                ocr_info
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                ocr_info
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)
        image_ids = kwargs.pop("image_ids", None)
        ocr_info = kwargs.pop("ocr_info", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

        if images is not None:
            _inputs["images"] = images
        if image_ids is not None:
            _inputs["image_ids"] = image_ids
        if ocr_info is not None:
            _inputs["ocr_info"] = ocr_info
        return _inputs