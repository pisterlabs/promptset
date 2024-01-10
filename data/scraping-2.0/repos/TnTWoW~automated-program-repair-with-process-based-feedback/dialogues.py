# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from huggingface_hub import ModelHubMixin, hf_hub_download

# Generic variable that is either ModelHubMixin or a subclass thereof
T = TypeVar("T", bound="ModelHubMixin")

TEMPLATE_FILENAME = "dialogue_template.json"
IGNORE_INDEX = -100


@dataclass
class DialogueTemplate(ModelHubMixin):
    """Converts all turns of a dialogue between a user and assistant to a standardized format.

    Adapted from OpenAI's ChatML (https://github.com/openai/openai-python/blob/main/chatml.md) and Vicuna (https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)
    """

    system: str
    messages: List[Dict[str, str]] = None
    system_token: str = "<|system|>"
    problem_token: str = "<|problem|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"

    def get_training_prompt(self) -> str:
        prompt = self.system_token + "\n" + self.system + self.end_token + "\n"
        if self.messages is None:
            raise ValueError("Dialogue template must have at least one message.")
        for message in self.messages:
            if message["role"] == "user":
                prompt += self.user_token + "\n" + message["content"] + self.end_token + "\n"
            else:
                prompt += self.assistant_token + "\n" + message["content"] + self.end_token + "\n"
        return prompt

    def get_inference_prompt(self) -> str:
        prompt = self.system_token + "\n" + self.system + self.end_token + "\n"
        if self.messages is None:
            raise ValueError("Dialogue template must have at least one message.")
        for message in self.messages:
            if message["role"] == "user":
                prompt += self.user_token + "\n" + message["content"] + self.end_token + "\n"
            else:
                prompt += self.assistant_token + "\n" + message["content"] + self.end_token + "\n"
        prompt += self.assistant_token
        return prompt

    def get_dialogue(self):
        """Helper function to format the messages as an easy-to-read dialogue."""
        prompt = ""
        if self.messages is None:
            raise ValueError("Dialogue template must have at least one message.")
        for message in self.messages:
            if message["role"] == "user":
                prompt += "\n\nHuman: " + message["content"]
            else:
                prompt += "\n\nAssistant: " + message["content"]
        return prompt

    def get_special_tokens(self) -> List[str]:
        return [self.system_token, self.user_token, self.assistant_token, self.end_token]

    def copy(self):
        return DialogueTemplate(
            system=self.system,
            messages=self.messages,
            system_token=self.system_token,
            user_token=self.user_token,
            assistant_token=self.assistant_token,
            end_token=self.end_token,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data):
        return DialogueTemplate(
            system=data["system"] if "system" in data else "",
            messages=data["messages"] if "messages" in data else None,
            system_token=data["system_token"] if "system_token" in data else "<|system|>",
            user_token=data["user_token"] if "user_token" in data else "<|user|>",
            assistant_token=data["assistant_token"] if "assistant_token" in data else "<|assistant|>",
            end_token=data["end_token"] if "end_token" in data else "<|end|>",
        )

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(exist_ok=True)
        with open(save_directory / "dialogue_template.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def _from_pretrained(
        cls: Type[T],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> T:
        """Loads the dialogue template from a local directory or the Huggingface Hub.

        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint (e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        if os.path.isdir(model_id):  # Can either be a local directory
            print("Loading dialogue template from local directory")
            template_file = os.path.join(model_id, TEMPLATE_FILENAME)
        else:  # Or a template on the Hub
            template_file = hf_hub_download(  # Download from the hub, passing same input args
                repo_id=model_id,
                filename=TEMPLATE_FILENAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        # Load template
        with open(template_file, "r") as f:
            data = json.load(f)
        return cls.from_dict(data=data)


# A shortened version of the system message in Anthropic's HHH prompt: https://gist.github.com/jareddk/2509330f8ef3d787fc5aaac67aab5f11#file-hhh_prompt-txt
default_template = DialogueTemplate(
    system="Below is a dialogue between a human user and an AI assistant. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed.",
)

# OpenAI and OpenAssistant train on few to no system messages.
# TODO: consider defining this as the `default` template
no_system_template = DialogueTemplate(
    system="",
)

alpaca_template = DialogueTemplate(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    user_token="### Instruction:",
    assistant_token="### Response:",
)

fix_template = DialogueTemplate(
    system="Below is a description and wrong answer for the programming problem. Write the correct solution to fix it.",
    problem_token="### Problem:",
    user_token="### Instruction:",
    assistant_token="### Response:",
)

SUPPORTED_DIALOGUE_TEMPLATES = {
    "default": default_template,
    "no_system": no_system_template,
    "alpaca": alpaca_template,
    "fix": fix_template,
}


def get_dialogue_template(template: str) -> DialogueTemplate:
    if template not in SUPPORTED_DIALOGUE_TEMPLATES.keys():
        raise ValueError(f"Template {template} is not supported!")
    return SUPPORTED_DIALOGUE_TEMPLATES[template].copy()


def prepare_dialogue(example, dialogue_template, is_train=True):
    if "wrong_code" in example.keys() and "acc_code" in example.keys() and example["wrong_code"] is not None and example["acc_code"] is not None:
        prompt = dialogue_template.system_token + "\n" + dialogue_template.system + dialogue_template.end_token + "\n"
        prompt += dialogue_template.user_token + "\n" + example["wrong_code"] + dialogue_template.end_token + "\n"
        if is_train:
            prompt += dialogue_template.assistant_token + "\n" + example[
                "acc_code"] + dialogue_template.end_token + "\n"
        else:
            prompt += dialogue_template.assistant_token
        example["text"] = prompt
    else:
        raise ValueError(
            f"Could not format example as dialogue! Require either `wrong_code` and `acc_code` or `wrong_code` and `acc_code` keys but found {list(example.keys())}"
        )
    return example

def prepare_dialogue_with_description(example, dialogue_template, is_train=True):
    if "wrong_code" in example.keys() and "acc_code" in example.keys() and example["wrong_code"] is not None and example["acc_code"] is not None:
        prompt = dialogue_template.system_token + "\n" + dialogue_template.system + dialogue_template.end_token + "\n"
        prompt += dialogue_template.problem_token + "\n"
        if not example["problem_description"]:
            print(example['problem_id'])
        prompt += example["problem_description"] + "\n" + dialogue_template.user_token + "\n" + example["wrong_code"] + dialogue_template.end_token + "\n"
        if is_train:
            prompt += dialogue_template.assistant_token + "\n" + example[
                "acc_code"] + dialogue_template.end_token + "\n"
        else:
            prompt += dialogue_template.assistant_token
        example["text"] = prompt
    else:
        raise ValueError(
            f"Could not format example as dialogue! Require either `wrong_code` and `acc_code` or `wrong_code` and `acc_code` keys but found {list(example.keys())}"
        )
    return example

def prepare_dialogue_with_description_multi_step(example, dialogue_template, is_train=True):
    if "codes" in example.keys():
        codes = eval(example['codes'])
        status = eval(example['status'])
        assert status != 'Accepted' and status[-1] == 'Accepted'
        prompt = dialogue_template.system_token + "\n" + dialogue_template.system + dialogue_template.end_token + "\n"
        prompt += dialogue_template.problem_token + "\n"
        if not example["problem_description"]:
            print('No description', example['problem_id'])
            sys.exit()
        prompt += example["problem_description"] + "\n" + dialogue_template.user_token + "\n" + codes[0].strip() + dialogue_template.end_token + "\n"
        if is_train:
            prompt += dialogue_template.assistant_token + "\n" + codes[-1].strip() + dialogue_template.end_token + "\n"
        else:
            prompt += dialogue_template.assistant_token
        example["text"] = prompt
    else:
        raise ValueError(
            f"Could not format example as dialogue! Require either `wrong_code` and `acc_code` or `wrong_code` and `acc_code` keys but found {list(example.keys())}"
        )
    return example


def mask_user_labels(tokenizer, dialogue_template, labels):
    """Masks the user turns of a dialogue from the loss"""
    user_token_id = tokenizer.convert_tokens_to_ids(dialogue_template.user_token)
    assistant_token_id = tokenizer.convert_tokens_to_ids(dialogue_template.assistant_token)
    for idx, label_id in enumerate(labels):
        if label_id == user_token_id:
            current_idx = idx
            while labels[current_idx] != assistant_token_id and current_idx < len(labels):
                labels[current_idx] = IGNORE_INDEX
                current_idx += 1
