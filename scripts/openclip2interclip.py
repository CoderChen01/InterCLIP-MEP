import dataclasses
import os
import re
import sys
from typing import Any, cast


import open_clip
from open_clip.hf_model import HFTextEncoder
from open_clip.model import CLIPVisionCfg
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel

from mmsd.interactive_clip import (
    CLIPConfig,
    CLIPPreTrainedModel,
    CLIPTextConfig,
    CLIPTextTransformer,
    CLIPVisionConfig,
    CLIPTextModel,
    CLIPVisionModel,
    CLIPVisionTransformer,
)

from mmsd.interactive_roberta import RobertaModel, RobertaConfig

VISION_CONFIG_MAP = {
    "layers": "num_hidden_layers",
    "width": "hidden_size",
    "patch_size": "patch_size",
    "image_size": "image_size",
}
STATE_DICT_PATTERNS = [
    # Vision
    (
        r"visual\.class_embedding",
        "vision_model.embeddings.class_embedding",
    ),
    (
        r"visual\.positional_embedding",
        "vision_model.embeddings.position_embedding.weight",
    ),
    (
        r"visual\.conv1\.(\w+)",
        "vision_model.embeddings.patch_embedding.{0}",
    ),
    (r"visual\.ln_pre\.(\w+)", "vision_model.pre_layrnorm.{0}"),
    (r"visual\.ln_post\.(\w+)", "vision_model.post_layernorm.{0}"),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.ln_1\.(\w+)",
        "vision_model.encoder.layers.{0}.layer_norm1.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.ln_2\.(\w+)",
        "vision_model.encoder.layers.{0}.layer_norm2.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.attn\.out_proj\.(\w+)",
        "vision_model.encoder.layers.{0}.self_attn.out_proj.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.mlp\.c_fc\.(\w+)",
        "vision_model.encoder.layers.{0}.mlp.fc1.{1}",
    ),
    (
        r"visual\.transformer\.resblocks\.(\w+)\.mlp\.c_proj\.(\w+)",
        "vision_model.encoder.layers.{0}.mlp.fc2.{1}",
    ),
    # Text
    (r"text\.transformer\.(.+)", "text_model.{0}"),
    (r"text\.proj\.(.+)", "text_projection.{0}"),
]


def convert_vision_config(config: CLIPVisionCfg) -> CLIPVisionConfig:
    conf = dataclasses.asdict(config)
    new_config: dict[str, Any] = {
        "hidden_act": "gelu",
    }
    for key, value in conf.items():
        if key in VISION_CONFIG_MAP:
            new_config[VISION_CONFIG_MAP[key]] = value
        elif key == "head_width":
            new_config["num_attention_heads"] = conf["width"] // value
        elif key == "mlp_ratio":
            new_config["intermediate_size"] = int(conf["width"] * value)
        elif not key.startswith("timm") and value:
            print(f"WARNING: Unknown key '{key}' in vision config.")

    return CLIPVisionConfig(**new_config)


def convert_state_dict(state_dict) -> dict[str, Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        found = False
        # special handling of vision attention blocks
        if match := re.match(
            r"visual\.transformer\.resblocks\.(\w+)\.attn\.in_proj_(\w+)", k
        ):
            # chunk weights into three
            chunks = v.chunk(3, dim=0)
            for proj_name, proj_v in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_k = f"vision_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                print(k, "--->", new_k)
                new_state_dict[new_k] = proj_v
                found = True
        # transpose visual projection
        elif k == "visual.proj":
            new_k = "visual_projection.weight"
            print(k, "--->", new_k)
            new_state_dict[new_k] = v.t()
            found = True
        else:
            for pattern, replacement in STATE_DICT_PATTERNS:
                if match := re.match(pattern, k):
                    new_k = replacement.format(*match.groups())
                    print(k, "--->", new_k)
                    new_state_dict[new_k] = v
                    found = True
                    break
        if not found:
            new_state_dict[k] = v

    return new_state_dict


if __name__ == "__main__":
    model_name = "roberta-ViT-B-32"
    pretrained = "laion2b_s12b_b32k"
    openclip_config: Any = open_clip.get_model_config(model_name)
    openclip_model: Any = open_clip.create_model(model_name, pretrained=pretrained)

    if not isinstance(openclip_model.text, HFTextEncoder):
        raise ValueError("Only HFTextEncoder is supported.")
    if openclip_config["text_cfg"]["hf_pooler_type"] != "mean_pooler":
        raise ValueError("Only mean_pooler is supported.")
    text_config = cast(PretrainedConfig, openclip_model.text.config)
    text_config.task_specific_params = {}
    vision_config = convert_vision_config(
        CLIPVisionCfg(**openclip_config["vision_cfg"])
    )
    vision_config.task_specific_params = {}

    state_dict = convert_state_dict(openclip_model.state_dict())

    text_model, tli = cast(
        tuple[PreTrainedModel, dict[str, Any]],
        CLIPTextModel.from_pretrained(
            None,
            config=RobertaConfig.from_dict(text_config.to_dict()),
            state_dict=state_dict,
            output_loading_info=True,
        ),
    )
    vision_model, vli = cast(
        tuple[PreTrainedModel, dict[str, Any]],
        CLIPVisionModel.from_pretrained(
            None,
            config=vision_config,
            state_dict=state_dict,
            output_loading_info=True,
        ),
    )

    text_model.save_pretrained("mmsd-results/clip-roberta/text")
    vision_model.save_pretrained("mmsd-results/clip-roberta/vision")