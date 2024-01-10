# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert BLIP-2 checkpoints from the original repository.

URL: https://github.com/salesforce/LAVIS/tree/main/projects/blip2
"""

import argparse
import json
from pathlib import Path
import shutil

import requests
import rich
import torch

# pip3 install salesforce-lavis
# I'm actually installing a slightly modified version: pip3 install git+https://github.com/nielsrogge/LAVIS.git@fix_lavis
from lavis.models import load_model_and_preprocess
from PIL import Image

from transformers import (
    AutoTokenizer,
    Blip2Processor,
    Blip2VisionConfig,
    BlipImageProcessor,
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from .configuration_blip2chatglm import Blip2ChatGLMConfig
from .modeling_blip2chatglm import Blip2ChatGLMForConditionalGeneration
from .configuration_chatglm import ChatGLMConfig


def load_demo_image():
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    return image


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # vision encoder
    rename_keys.append(("visual_encoder.cls_token", "vision_model.embeddings.class_embedding"))
    rename_keys.append(("visual_encoder.pos_embed", "vision_model.embeddings.position_embedding"))
    rename_keys.append(("visual_encoder.patch_embed.proj.weight", "vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("visual_encoder.patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.bias"))
    rename_keys.append(("ln_vision.weight", "vision_model.post_layernorm.weight"))
    rename_keys.append(("ln_vision.bias", "vision_model.post_layernorm.bias"))

    for i in range(config.vision_config.num_hidden_layers):
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.weight", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.weight", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.qkv.weight", f"vision_model.encoder.layers.{i}.self_attn.qkv.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.weight", f"vision_model.encoder.layers.{i}.self_attn.projection.weight",))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.bias", f"vision_model.encoder.layers.{i}.self_attn.projection.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.weight", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.weight", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))

    # QFormer
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.weight", "qformer.layernorm.weight"))
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.bias", "qformer.layernorm.bias"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def read_in_q_v_bias(state_dict, config):
    for i in range(config.vision_config.num_hidden_layers):
        # read in original q and v biases
        q_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.v_bias")

        # next, set bias in the state dict
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)
        )
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias


def get_blip2_config(model_name):
    image_size = 224
    vision_config = Blip2VisionConfig(
        image_size=image_size, torch_dtype="float16"
    ).to_dict()

    # make sure the models have proper bos_token_id and eos_token_id set (important for generation)
    # seems like flan-T5 models don't have bos_token_id properly set?
    if "chatglm-6b" in model_name:
        text_config = ChatGLMConfig.from_pretrained(
            "/home/wsh/models/chatglm-6b"
        ).to_dict()

    config = Blip2ChatGLMConfig(vision_config=vision_config, text_config=text_config)

    return config, image_size


def count_parameters(model):
    from rich.table import Table

    table = Table(title="Params")
    table.add_column("Name", style="dim", no_wrap=True)
    table.add_column("Params", justify="right")
    total_params = 0

    for name, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row(name, str(params))
        total_params += params

    return table, total_params


@torch.no_grad()
def convert_blip2zh_checkpoint(
    model_name, pytorch_dump_folder_path=None, push_to_hub=False
):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    if "chatglm-6b" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/wsh/models/chatglm-6b", trust_remote_code=True
        )
    else:
        raise NotImplementedError()

    config, image_size = get_blip2_config(model_name)

    model_name_to_original = {
        "blip2zh-chatglm-6b": ("blip2zh_chatglm", "pretrain_chatglm6b"),
        "blip2zh-chatglm-6b-vqa": ("blip2zh_chatglm", "vqa"),
    }

    name, type = model_name_to_original[model_name]

    # load original model
    print("Loading original model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model, vis_processors, _ = load_model_and_preprocess(
        name=name, model_type=type, is_eval=True, device=device
    )
    original_model.eval()
    print("Done!")

    # update state dict keys
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # some keys can be renamed efficiently
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("Qformer.bert"):
            key = key.replace("Qformer.bert", "qformer")
        if "attention.self" in key:
            key = key.replace("self", "attention")
        if key.startswith("lm_proj"):
            key = key.replace("lm_proj", "language_projection")
        if key.startswith("lm_model"):
            key = key.replace("lm_model", "language_model")
        state_dict[key] = val

    # read in qv biases
    read_in_q_v_bias(state_dict, config)

    hf_model = Blip2ChatGLMForConditionalGeneration(config)
    hf_model.setup_dtype(vision_encoder_dtype="fp16", lm_dtype="fp16")
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    assert len(missing_keys) == 0, missing_keys
    for unexpected_key in unexpected_keys:
        if unexpected_key == "qformer.embeddings.position_ids":
            pass
        else:
            raise ValueError(f"Unexpected key: {unexpected_key}")

    image = load_demo_image()
    original_pixel_values = vis_processors["eval"](image).unsqueeze(0).to(device)

    # create processor
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size},
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # make sure processor creates exact same pixel values
    assert torch.allclose(pixel_values, original_pixel_values)

    original_model.to(device)
    hf_model.eval()
    hf_model.to(device)
    # since lm is freezed, we only assert vtokens are all close
    with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
        original_logits = original_model(
            {"image": original_pixel_values, "text_input": [""]}
        )["logits"]
        a_ids = tokenizer.encode("", add_special_tokens=False)
        b_ids = tokenizer.encode("", add_special_tokens=False)
        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        input_ids = torch.cat(
            [
                torch.ones(
                    (1, hf_model.config.num_query_tokens),
                    dtype=input_ids.dtype,
                    device=device,
                )
                * tokenizer.unk_token_id,
                input_ids,
            ],
            dim=1,
        )
        logits = hf_model(pixel_values, input_ids).logits

    assert original_logits.shape == logits.shape, f"{original_logits.shape} vs {logits.shape}"
    print("First values of original logits:", original_logits[0, :3, :3])
    print("First values of HF logits:", logits[0, :3, :3])
    print("Last values of original logits:", original_logits[0, -3:, -3:])
    print("Last values of HF logits:", logits[0, -3:, -3:])

    # assert values
    target_dtype = logits.dtype
    assert torch.allclose(original_logits.to(target_dtype), logits, atol=5e-2)
    print("Looks ok!")

    print("Generating a caption...")
    with torch.cuda.amp.autocast(enabled=True):
        #    for response, history in original_model.generate("这是一张什么图片？", history=[], max_length=128):
        #        print(response)
        for response in hf_model.stream_chat(
            tokenizer, ("描述一下这张图", original_pixel_values), history=[]
        ):
            print(response)

    if pytorch_dump_folder_path is not None:
        # store chatglm tokenizer
        processor.save_pretrained(pytorch_dump_folder_path)
        # store Blip2ChatGLM
        hf_model.save_pretrained(pytorch_dump_folder_path, max_shard_size="2GB")

    pytorch_dump_folder_path = Path(pytorch_dump_folder_path)
    with (pytorch_dump_folder_path / "config.json").open("r", encoding="utf8") as rf:
        config = json.load(rf)
    config["auto_map"] = {
        "AutoConfig": "configuration_blip2chatglm.Blip2ChatGLMConfig",
        "AutoModel": "modeling_blip2chatglm.Blip2ChatGLMForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_blip2chatglm.Blip2ChatGLMForConditionalGeneration",
    }
    with (pytorch_dump_folder_path / "config.json").open("w", encoding="utf8") as wf:
        json.dump(config, wf, indent=2, ensure_ascii=False)
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/configuration_blip2chatglm.py",
        pytorch_dump_folder_path / "configuration_blip2chatglm.py",
    )
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/modeling_blip2chatglm.py",
        pytorch_dump_folder_path / "modeling_blip2chatglm.py",
    )
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/configuration_chatglm.py",
        pytorch_dump_folder_path / "configuration_chatglm.py",
    )
    shutil.copy(
        "lavis/models/blip2zh_models/blip2zh_chatglm/modeling_chatglm.py",
        pytorch_dump_folder_path / "modeling_chatglm.py",
    )

    params, total_size = count_parameters(hf_model)
    # rich.print(params)
    rich.print(f"Total size: {total_size}")

    # if push_to_hub:
    #     processor.push_to_hub(f"nielsr/{model_name}")
    #     hf_model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = [
        "blip2zh-chatglm-6b",
        "blip2zh-chatglm-6b-vqa",
    ]
    parser.add_argument(
        "--model_name",
        default="blip2zh-chatglm-6b",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_blip2zh_checkpoint(
        args.model_name, args.pytorch_dump_folder_path, args.push_to_hub
    )
