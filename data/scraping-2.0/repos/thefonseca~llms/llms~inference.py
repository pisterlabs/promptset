import logging
import os
from pprint import pformat
import re
import time

import fire
import numpy as np

from .models.huggingface import (
    Text2TextLM,
    CausalLM,
    LlamaChat,
    InstructText2TextLM,
    InstructCausalLM,
    Alpaca,
    Vicuna,
)
from .models.openai import OpenAIChat
from .utils.utils import get_progress_bar, add_progress_task

logger = logging.getLogger(__name__)

MODEL_CACHE = {}

DEFAULT_MODEL_MAP = {
    "gpt-[-\d\w]*": OpenAIChat,
    "facebook/opt-[\d\w]+": CausalLM,
    ".*[cC]ode[Ll]lama-[\d\w]+-[Ii]nstruct-hf": LlamaChat,
    ".*llama-?2.*chat.*": LlamaChat,
    ".*[Ll]lama.*": CausalLM,
    "bigscience/T0[_\d\w]*": InstructText2TextLM,
    "google/flan-t5[-\d\w]+": InstructText2TextLM,
    "google/long-t5[-\d\w]+": InstructText2TextLM,
    ".*alpaca.*": Alpaca,
    ".*vicuna.*": Vicuna,
    "mosaicml/mpt[-\d\w]$": CausalLM,
    "tiiuae/falcon[-\d\w]$": CausalLM,
    "mosaicml/mpt[-\d\w]+instruct": Alpaca,
    "tiiuae/falcon[-\d\w]+instruct": InstructCausalLM,
}


def parse_kwargs(kwargs, model_prefix="model_"):
    model_kwargs = {}
    generation_kwargs = {}

    for key, value in kwargs.items():
        if key[: len(model_prefix)] == model_prefix:
            key = key[len(model_prefix) :]
            model_kwargs[key] = value
        else:
            generation_kwargs[key] = value

    return model_kwargs, generation_kwargs


def get_model_class(model_name, model_map=None, default_class=None):
    if model_map is None:
        model_map = DEFAULT_MODEL_MAP

    for key, val in model_map.items():
        if re.match(key, model_name):
            summarizer_class = val
            break
    else:
        logger.warning(f"Could not match model '{model_name}' to generator class")
        if default_class:
            summarizer_class = default_class
        else:
            summarizer_class = Text2TextLM

    return summarizer_class


def get_sample_gen_kwargs(kwargs, sample_idx):
    sample_kwargs = {}
    for arg_name, arg_val in kwargs.items():
        if arg_val is not None and callable(arg_val):
            sample_kwargs[arg_name] = arg_val(sample_idx)
        elif arg_val is not None and isinstance(arg_val, list):
            sample_kwargs[arg_name] = arg_val[sample_idx]
        else:
            sample_kwargs[arg_name] = arg_val
    return sample_kwargs


def token_statistics(
        model,
        inputs,
        truncation=True,
        show_progress=True,
        **generation_kwargs,
    ):
        progress = get_progress_bar()
        task = add_progress_task(
            progress,
            f"Calculating token statistics for {model.model_name}...",
            total=len(inputs),
            existing_ok=False,
        )
        progress.update(task, visible=show_progress)
        truncated_tokens = []
        num_tokens = []

        with progress:
            for idx, input_data in enumerate(inputs):
                sample_kwargs = get_sample_gen_kwargs(generation_kwargs, idx)
                result = model.token_statistics(
                    input_data, truncation, **sample_kwargs
                )
                num_tokens.append(result[0])
                truncated_tokens.append(result[1])
                progress.update(task, advance=1)

        stats = dict(
            total_tokens=sum(num_tokens),
            mean_tokens=np.mean(num_tokens),
            total_truncation=sum(truncated_tokens),
            mean_truncation=np.mean(truncated_tokens),
        )
        return stats


def generate(
    model_name,
    sources=None,
    model_class=None,
    max_length=256,
    cache_start=0,
    cache_end=None,
    use_model_cache=False,
    ignore_errors=False,
    show_progress=True,
    **kwargs,
):
    outputs = []
    progress = get_progress_bar()
    if sources is None:
        sources = [None]
        show_progress = False

    task = add_progress_task(
        progress,
        f"Generating outputs for {model_name}...",
        total=len(sources),
        existing_ok=False,
    )
    progress.update(task, visible=show_progress)
    cache_end = cache_end if cache_end is not None else len(sources)
    model_kwargs, generation_kwargs = parse_kwargs(kwargs)

    if model_class is None:
        model_class = get_model_class(model_name)

    logger.info(f"Using model: {model_class}")
    model = model_class(model_name, **model_kwargs)

    if use_model_cache:
        cached_model = MODEL_CACHE.get(model_name)
        if cached_model and hasattr(cached_model, "model"):
            model.model = cached_model.model
        MODEL_CACHE[model_name] = model

    if hasattr(model, "token_statistics"):
        stats = token_statistics(
            model,
            sources,
            max_length=max_length,
            show_progress=show_progress,
            **generation_kwargs,
        )
        logger.info(f"Token statistics for input:\n{pformat(stats)}")

    with progress:
        for idx, text in enumerate(sources):
            sample_kwargs = get_sample_gen_kwargs(generation_kwargs, idx)
            ignore_cache = idx < cache_start or idx >= cache_end
            try:
                output = model.generate(
                    text,
                    max_length=max_length,
                    memoizer_ignore_cache=ignore_cache,
                    verbose=idx == 0,
                    **sample_kwargs,
                )
            except Exception as err:
                logger.error(f"Generation failed for sample {idx}")
                if ignore_errors:
                    logger.error(err)
                    output = None
                else:
                    raise err

            outputs.append(output)
            progress.update(task, advance=1)

            is_cache_hit = model.is_last_result_from_cache()
            if (
                hasattr(model, "request_interval")
                and model.request_interval > 0
                and not is_cache_hit
            ):
                time.sleep(model.request_interval)

    return outputs


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(generate)