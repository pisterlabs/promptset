# Importing the necessary libraries
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers, HuggingFacePipeline
from langchain.llms.huggingface_hub import HuggingFaceHub
from loguru import logger as log
from transformers import AutoTokenizer, TextStreamer, pipeline

from utils.helpers import get_config

# Get the configuration
cfg = get_config('language_models.yaml')

# %%: Device to use
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %%: Create a type alias for the language models
LanguageModel = HuggingFacePipeline | CTransformers | ChatOpenAI


# %%: Get language models
def get_huggingface_model() -> HuggingFaceHub:
    """Get the HuggingFace model from the HuggingFace Hub"""
    # TODO: Warn if the user didn't enter a api_token
    return HuggingFaceHub(repo_id=cfg.huggingface_model.repo_id, model_kwargs=cfg.huggingface_model.model_config)


def get_openai_model() -> ChatOpenAI:
    """Get the OpenAI model from the OpenAI API"""
    return ChatOpenAI(config=cfg.model_config, do_sample=cfg.do_sample)


def get_mistral_model() -> CTransformers:
    """Get the Mistral model with CTransformers"""
    return CTransformers(model=cfg.mistral_model.path, model_type='mistral', device=DEVICE, do_sample=cfg.do_sample,
                         config=cfg.model_config)


def get_gguf_model() -> CTransformers:
    """Get the GGUF model with CTransformers"""
    return CTransformers(model=cfg.gguf_model.path, model_type=cfg.gguf_model.type, device=DEVICE,
                         do_sample=cfg.do_sample, config=cfg.model_config)


def get_gptq_model() -> HuggingFacePipeline:
    """Get the GPTQ model with AutoGPTQForCausalLM"""
    model = AutoGPTQForCausalLM.from_quantized(cfg.gptq_model.model_name, revision="main",
                                               model_basename=cfg.gptq_model.model_basename,
                                               use_safetensors=cfg.gptq_model.use_safetensors,
                                               trust_remote_code=cfg.gptq_model.trust_remote_code,
                                               inject_fused_attention=cfg.gptq_model.inject_fused_attention,
                                               device=DEVICE,
                                               quantize_config=None)

    tokenizer = AutoTokenizer.from_pretrained(cfg.gptq_model.model_name,
                                              use_fast=cfg.gptq_model.use_fast)

    streamer = TextStreamer(tokenizer, skip_prompt=cfg.gptq_streamer.skip_prompt,
                            skip_special_tokens=cfg.gptq_streamer.skip_special_tokens)

    text_pipeline = pipeline("text2text-generation",
                             model=model,
                             tokenizer=tokenizer,
                             max_new_tokens=cfg.model_config.max_new_tokens,
                             temperature=cfg.model_config.temperature,
                             top_p=cfg.model_config.top_p,
                             do_sample=cfg.do_sample,
                             repetition_penalty=cfg.model_config.repetition_penalty,
                             streamer=streamer)

    return HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": cfg.model_config.temperature})


def get_language_model(model_name: str) -> LanguageModel | None:
    """
    Get a language model based on the given name.

    Parameters
    ----------
    model_name: str
        The name of the model to get.

    Returns
    -------
    HuggingFacePipeline, CTransformers, or ChatOpenAI instance:
        The loaded language model.

    Warnings
    - log (loguru): If the model name is unknown.
    """
    mapper: dict[str, LanguageModel] = {
        'Llama-2-7B GPTQ (GPU)': get_gptq_model,
        'Llama-2-13B GGUF (CPU)': get_gguf_model,
        'HuggingFace API (Online)': get_huggingface_model,
        'OpenAI API (Online)': get_openai_model,
        'Zephyr-7B (CPU)': get_mistral_model,
    }

    if model_name in mapper:
        return mapper[model_name]()
    else:
        log.error(f'Unknown model name: {model_name}')
        return None
