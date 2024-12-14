import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from mmsd.interactive_clip import CLIPTextModel, CLIPVisionModel
from mmsd.interactive_roberta import RobertaConfig

logger = logging.getLogger("mmsd.model")


@dataclass
class MMSDOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    fused_embeds: Optional[torch.Tensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            (
                self[k]
                if k not in ["text_model_output", "vision_model_output"]
                else getattr(self, k).to_tuple()
            )
            for k in self.keys()
        )


class MMSDConfig(PretrainedConfig):

    def __init__(
        self,
        clip_vision_model: Optional[str] = None,
        clip_text_model: Optional[str] = None,
        is_openclip: bool = False,
        vision_conditional_layer_ids: Optional[list[int]] = None,
        text_conditional_layer_ids: Optional[list[int]] = None,
        vision_embed_dim: Optional[int] = None,
        text_embed_dim: Optional[int] = None,
        is_v2t_adapter_mlp: bool = True,
        is_t2v_adapter_mlp: bool = True,
        projection_dim: int = 1024,
        use_sim_loss: bool = True,
        **kwargs,
    ) -> None:
        kwargs.pop("vision_task_params", None)
        kwargs.pop("text_task_params", None)

        super().__init__(
            id2label={
                0: "non_sarcastic",
                1: "sarcastic",
            },
            lable2id={
                "non_sarcastic": 0,
                "sarcastic": 1,
            },
            **kwargs,
        )

        if clip_vision_model is None:
            raise ValueError("clip_vision_model should be provided.")
        if clip_text_model is None:
            raise ValueError("clip_text_model should be provided.")
        if vision_embed_dim is None or text_embed_dim is None:
            raise ValueError("vision_embed_dim and text_embed_dim should be provided.")

        if (
            vision_conditional_layer_ids is not None
            and len(vision_conditional_layer_ids) > 0
        ):
            self.vision_task_params = {
                "cond_hidden_size": text_embed_dim,
                "is_conditional": True,
                "cond_attn_layer_inds": vision_conditional_layer_ids,
                "is_mlp": is_t2v_adapter_mlp,
            }
        else:
            self.vision_task_params = {
                "cond_hidden_size": text_embed_dim,
                "is_conditional": False,
                "cond_attn_layer_inds": [],
            }

        if (
            text_conditional_layer_ids is not None
            and len(text_conditional_layer_ids) > 0
        ):
            self.text_task_params = {
                "cond_hidden_size": vision_embed_dim,
                "is_conditional": True,
                "cond_attn_layer_inds": text_conditional_layer_ids,
                "is_mlp": is_v2t_adapter_mlp,
            }
        else:
            self.text_task_params = {
                "cond_hidden_size": vision_embed_dim,
                "is_conditional": False,
                "cond_attn_layer_inds": [],
            }

        self.is_openclip = is_openclip
        self.clip_vision_model = clip_vision_model
        self.clip_text_model = clip_text_model

        self.vision_conditional_layer_ids = vision_conditional_layer_ids
        self.text_conditional_layer_ids = text_conditional_layer_ids

        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim

        self.is_v2t_adapter_mlp = is_v2t_adapter_mlp
        self.is_t2v_adapter_mlp = is_t2v_adapter_mlp

        self.projection_dim = projection_dim
        self.use_sim_loss = use_sim_loss


def cosine_similarity_loss(
    fused_embeds: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    assert (label == 1).size(0) > 0 or (label == 0).size(0) > 0
    p_embeds = fused_embeds[label == 1]
    n_embeds = fused_embeds[label == 0]
    sim_loss = torch.tensor(0.0, device=fused_embeds.device)
    if (label == 1).size(0) > 0:
        sim_loss += (1 - p_embeds @ p_embeds.t()).mean()
    if (label == 0).size(0) > 0:
        sim_loss += (1 - n_embeds @ n_embeds.t()).mean()
        tmp = p_embeds @ n_embeds.t()
        sim_loss += (torch.maximum(torch.zeros_like(tmp), tmp)).mean()
    return sim_loss


class InteractiveCLIP4MMSD(PreTrainedModel):
    config_class = MMSDConfig
    base_model_prefix = "mmsd"
    supports_gradient_checkpointing = True

    def __init__(self, config: MMSDConfig) -> None:
        super().__init__(config)
        self.config = config

        self._init_clip()

        if self.config.use_sim_loss:
            self.fuse_projection = nn.Sequential(
                nn.Linear(
                    self.config.text_embed_dim + self.config.vision_embed_dim,
                    self.config.projection_dim,
                ),
                nn.ReLU(),
                nn.Linear(self.config.projection_dim, self.config.projection_dim),
            )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.text_embed_dim + self.config.vision_embed_dim,
                self.config.projection_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.num_labels),
        )

        self.post_init()

    def _init_weights(self, module) -> None:
        fc1_std = (self.config.text_embed_dim + self.config.vision_embed_dim) ** -0.5
        fc2_std = self.config.projection_dim**-0.5

        if hasattr(module, "fuse_projection"):
            nn.init.normal_(module.fuse_projection[0].weight, std=fc1_std)
            nn.init.zeros_(module.fuse_projection[0].bias)
            nn.init.normal_(module.fuse_projection[2].weight, std=fc2_std)
            nn.init.zeros_(module.fuse_projection[2].bias)
        if hasattr(module, "classifier"):
            nn.init.normal_(module.classifier[0].weight, std=fc1_std)
            nn.init.zeros_(module.classifier[0].bias)
            nn.init.normal_(module.classifier[2].weight, std=fc2_std)
            nn.init.zeros_(module.classifier[2].bias)

    def _init_clip(self) -> None:
        if not self.config.is_openclip:
            self.vision_model = cast(
                CLIPVisionModel,
                CLIPVisionModel.from_pretrained(
                    self.config.clip_vision_model,
                    task_specific_params=self.config.vision_task_params,
                ),
            )

            self.text_model = cast(
                CLIPTextModel,
                CLIPTextModel.from_pretrained(
                    self.config.clip_text_model,
                    task_specific_params=self.config.text_task_params,
                ),
            )
        else:
            self.vision_model = cast(
                CLIPVisionModel,
                CLIPVisionModel.from_pretrained(
                    self.config.clip_vision_model,
                    task_specific_params=self.config.vision_task_params,
                ),
            )

            # TODO(chenjunjie): this is a temporary solution
            config_path = Path(self.config.clip_text_model) / "config.json"
            config = RobertaConfig.from_json_file(config_path)
            config.task_specific_params = self.config.text_task_params
            self.text_model = cast(
                CLIPTextModel,
                CLIPTextModel.from_pretrained(
                    self.config.clip_text_model,
                    config=config,
                ),
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MMSDOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            conditional_hidden_states=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            conditional_hidden_states=None,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.text_model.config.task_specific_params["is_conditional"]:
            text_st_image_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                conditional_hidden_states=vision_outputs[0].detach(),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            text_st_image_outputs = text_outputs

        if self.vision_model.config.task_specific_params["is_conditional"]:
            image_st_text_outputs = self.vision_model(
                pixel_values=pixel_values,
                conditional_hidden_states=text_outputs[0].detach(),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            image_st_text_outputs = vision_outputs

        image_st_text_embeds = image_st_text_outputs[1]
        text_st_image_embeds = text_st_image_outputs[1]

        fused_embeds = torch.cat([image_st_text_embeds, text_st_image_embeds], dim=-1)
        logits = self.classifier(fused_embeds)

        if self.config.use_sim_loss:
            fused_embeds = self.fuse_projection(fused_embeds)
            fused_embeds = fused_embeds / fused_embeds.norm(dim=-1, p=2, keepdim=True)

        loss = None
        if return_loss is None or return_loss:
            if label is not None:
                loss = torch.nn.functional.cross_entropy(logits, label)
                if self.config.use_sim_loss:
                    loss += cosine_similarity_loss(fused_embeds, label)

        if not return_dict:
            output = (
                logits,
                fused_embeds,
            )
            return ((loss,) + output) if loss is not None else output

        return MMSDOutput(
            loss=loss,
            logits=logits,
            fused_embeds=fused_embeds,
        )
