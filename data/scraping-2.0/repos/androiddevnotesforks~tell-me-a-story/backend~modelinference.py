# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University
# Authors, the HuggingFace Inc. team., and Randy West
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
#
# This file is more or less
# https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
# refactored as a library API
""" Conditional text generation with the auto-regressive models of the library
(GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange
from time import time

import torch
import torch.nn.functional as F
import numpy as np

from transformers import (
    GPT2Config,
    OpenAIGPTConfig,
    XLNetConfig,
    TransfoXLConfig,
    XLMConfig,
    CTRLConfig,
)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            GPT2Config,
            OpenAIGPTConfig,
            XLNetConfig,
            TransfoXLConfig,
            XLMConfig,
            CTRLConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


class ModelInference:
    """
    Inference entrypoint

    Keyword arguments:
    model_type -- Model type selected (see MODEL_CLASSES)
    model_name_or_path -- Path to pre-trained model or shortcut name (see ALL_MODELS)
    padding_text -- Padding text to long memory models (TransfoXL). Default if unspecified
    xlm_lang -- Optional language when used with the XLM model
    num_samples -- Number of outputs to produce
    temperature -- sampling temperature. 0 implies greedy
    repetition_penalty -- primarily useful for CTRL model; in that case, use 1.2
    top_k -- keep top k logits to sample from at each step
    top_p -- keep top logits until cumulative mass exceeds this
    no_cuda -- Avoid using CUDA when available
    seed -- random seed for initialization
    stop_token -- Token at which text generation is stopped
    progress_bar -- If true, display a progress bar while sampling. Turn off for better logging
    """

    def __init__(
        self,
        model_type,
        model_name_or_path,
        padding_text="",
        xlm_lang="",
        num_samples=1,
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=0.9,
        no_cuda=None,
        seed=42,
        stop_token=None,
        progress_bar=True,
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.padding_text = padding_text
        self.xlm_lang = xlm_lang
        self.num_samples = num_samples
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.no_cuda = no_cuda
        self.seed = seed
        self.stop_token = stop_token
        self.progress_bar = progress_bar

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 3.5
            and not self.no_cuda
            else "cpu"
        )
        n_gpu = torch.cuda.device_count()

        self._set_seed(n_gpu)

        self.model_type = self.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)
        self.model = model_class.from_pretrained(self.model_name_or_path)
        print(f"Device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

        if self.model_type in ["ctrl"]:
            if self.temperature > 0.7:
                logger.info(
                    "CTRL typically works better with lower temperatures (and lower top_k)."
                )

        # XLM Language usage detailed in the issues #1414
        xlm_lang = None
        if (
            self.model_type in ["xlm"]
            and hasattr(self.tokenizer, "lang2id")
            and hasattr(self.model.config, "use_lang_emb")
            and self.model.config.use_lang_emb
        ):
            if self.xlm_lang not in self.tokenizer.lang2id.keys():
                raise ValueError(
                    f"self.xlm_lang ({self.xlm_lang}) not in tokenizer langs ({self.tokenizer.lang2id.keys()})"
                )
            xlm_lang = self.tokenizer.lang2id[self.xlm_lang]
        self.xlm_lang = xlm_lang

        # XLM masked-language modeling (MLM) models need masked token (see details in _sample_sequence)
        self.is_xlm_mlm = (
            self.model_type in ["xlm"] and "mlm" in self.model_name_or_path
        )
        if self.is_xlm_mlm:
            self.xlm_mask_token = self.tokenizer.mask_token_id
        else:
            self.xlm_mask_token = None

    def sample_and_decode(self, prompt, length=20):
        """
        Sample from the model using prompt (and also padding_text, specified in
        __init__, for transfo-xl and xlnet) as context

        Keyword arguments:
        prompt -- the model prompt
        length -- Output max length
        """

        if length < 0 and self.model.config.max_position_embeddings > 0:
            length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < length:
            length = (
                self.model.config.max_position_embeddings
            )  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop

        raw_text = prompt
        if self.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (
                self.padding_text if self.padding_text else PADDING_TEXT
            ) + raw_text
        context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)
        if self.model_type == "ctrl":
            if not any(
                context_tokens[0] == x for x in self.tokenizer.control_codes.values()
            ):
                logger.info(
                    "WARNING! You are not starting your generation from a control code so you won't get good results"
                )

        out = self._sample_sequence(context_tokens, length)
        out = out[:, len(context_tokens) :].tolist()
        all_text = []
        for o in out:
            text = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(self.stop_token) if self.stop_token else None]
            all_text.append(text)

        return all_text

    def _set_seed(self, n_gpu):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def _top_k_top_p_filtering(self, logits, filter_value=-float("Inf")):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size x vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        self.top_k = min(self.top_k, logits.size(-1))  # Safety check
        if self.top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = (
                logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            )
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        return logits

    def _sample_sequence(self, context, length):
        is_xlnet = bool(self.model_type == "xlnet")
        use_past = self.model_type in ("gpt2", "ctrl", "transfo-xl")
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0).repeat(self.num_samples, 1)
        generated = context
        outputs = None
        range_fn = trange if self.progress_bar else range
        if self.progress_bar:
            range_fn = trange
        else:
            range_fn = range
            print("Sampling has begun")
            start_time = time()
        with torch.no_grad():
            for _ in range_fn(length):

                if use_past and outputs is not None:
                    inputs = {"input_ids": generated[:, -1:], "past": outputs[1]}
                else:
                    inputs = {"input_ids": generated}
                if is_xlnet:
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat(
                        (
                            generated,
                            torch.zeros((1, 1), dtype=torch.long, device=self.device),
                        ),
                        dim=1,
                    )
                    perm_mask = torch.zeros(
                        (1, input_ids.shape[1], input_ids.shape[1]),
                        dtype=torch.float,
                        device=self.device,
                    )
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros(
                        (1, 1, input_ids.shape[1]),
                        dtype=torch.float,
                        device=self.device,
                    )
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {
                        "input_ids": input_ids,
                        "perm_mask": perm_mask,
                        "target_mapping": target_mapping,
                    }

                if self.is_xlm_mlm and self.xlm_mask_token:
                    # XLM MLM models are direct models (predict same token, not next token)
                    # => need one additional dummy token in the input (will be masked and guessed)
                    input_ids = torch.cat(
                        (
                            generated,
                            torch.full(
                                (1, 1),
                                self.xlm_mask_token,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ),
                        dim=1,
                    )
                    inputs = {"input_ids": input_ids}

                if self.xlm_lang is not None:
                    inputs["langs"] = torch.tensor(
                        [self.xlm_lang] * inputs["input_ids"].shape[1],
                        device=self.device,
                    ).view(1, -1)

                outputs = self.model(
                    **inputs
                )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][:, -1, :] / (
                    self.temperature if self.temperature > 0 else 1.0
                )

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for i in range(self.num_samples):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= self.repetition_penalty

                filtered_logits = self._top_k_top_p_filtering(next_token_logits)
                if self.temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(
                        F.softmax(filtered_logits, dim=-1), num_samples=1
                    )
                generated = torch.cat((generated, next_token), dim=1)
        if not self.progress_bar:
            print(f"Sampling completed in {time()-start_time:.2f} seconds")
        return generated

