from typing import List, Optional, Tuple

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3Config, Qwen3ForCausalLM

from ..vivqa_arch import ViVQAMetaForCausalLM, ViVQAMetaModel


class ViVQAConfig(Qwen3Config):
    model_type = "vivqa"


class ViVQAModel(Qwen3Model, ViVQAMetaModel):
    """Backbone Qwen3 + module vision (SigLIP + projector)."""
    config_class = ViVQAConfig

    def __init__(self, config: Qwen3Config):
        Qwen3Model.__init__(self, config)
        ViVQAMetaModel.__init__(self, config)


class ViVQAForCausalLM(Qwen3ForCausalLM, ViVQAMetaForCausalLM):

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
            image_ids: Optional[torch.LongTensor] = None,
            **kwargs,
            ) -> Tuple:
        assert image_ids is not None, "image_ids none at ViVQAForCausalLM."
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_ids
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
        return _inputs