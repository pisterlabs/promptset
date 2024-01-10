import logging
from typing import Any
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
)
from langchain.llms import HuggingFacePipeline

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_pretrained_model(model_path: str, device_map: Any):
    if device_map is None:
        device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        local_files_only=True,
    )
    return model


def load_pretrained_tokenizer(model_path: str):
    return transformers.AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
    )


def create_nf4_model_config():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return nf4_config


def create_llm_pipe(llm_model, tokenizer, **kwargs):
    new_kwargs = dict(kwargs)
    new_kwargs['max_new_tokens'] = kwargs.get('max_new_tokens', 1024)
    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        **new_kwargs
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf
