import logging
import importlib.util
import os
from typing import List, Union

from parrot1 import PARROT_CACHED_MODELS
from parrot1.audio.transcription.model import TimedTranscription

from parrot1.commons.generative.base import BaseLLMModel
from parrot1.commons.generative.llamacpp import LlamaCppModel
from parrot1.commons.generative.openai_gpt import OpenaiGPTModel
from parrot1.config.config import PARROT_CONFIGS

from parrot1.recap.tasks import ParrotTask, resolve_prompt_from_task

imp_llama_cpp = importlib.util.find_spec(name="llama_cpp")

has_llama_cpp = imp_llama_cpp is not None

__logger = logging.getLogger(__name__)


def get_client(use_llama_cpp: bool = False) -> Union[BaseLLMModel, None]:
    if not use_llama_cpp:
        if os.getenv("OPENAI_API_KEY") is not None:
            return OpenaiGPTModel(
                model_size_or_type=PARROT_CONFIGS.parrot_configs.generative_models.openai.type_or_size
            )
        else:
            __logger.error(
                "OPENAI_API_KEY is not set but you're trying to use the OpenAI Apis."
            )
            return None
    else:
        if has_llama_cpp:
            cache_root = PARROT_CACHED_MODELS
            __logger.info("Using llama_cpp model")
            __logger.info(f"Using cache folder at {cache_root.as_posix()}")
            os.makedirs(cache_root, exist_ok=True)
            return LlamaCppModel(
                repo_id=PARROT_CONFIGS.parrot_configs.generative_models.llama_cpp.repo_id,
                model_size_or_type=PARROT_CONFIGS.parrot_configs.generative_models.llama_cpp.type_or_size,
            )
        else:
            __logger.error(
                "The llama-cpp-python package was not installed. Try fixing it by doing pip install parrot1[llama-cpp]."
            )
            return None


async def generate_chunks(client: BaseLLMModel, texts: List[str]) -> List[str]:
    prompt = resolve_prompt_from_task(
        ParrotTask.CHUNK, language=PARROT_CONFIGS.parrot_configs.language
    )

    summaries = await client.generate_from_prompts(
        prompts=[prompt.format(text=text) for text in texts],
        max_tokens=PARROT_CONFIGS.parrot_configs.generative_models.chunking.max_tokens,
        temperature=PARROT_CONFIGS.parrot_configs.generative_models.chunking.temperature,
    )

    return summaries


async def generate_final_result(
    texts: List[TimedTranscription],
    task: ParrotTask = ParrotTask.RECAP,
    use_llama_cpp: bool = False,
) -> str:
    prompt = resolve_prompt_from_task(
        task, language=PARROT_CONFIGS.parrot_configs.language
    )
    client = get_client(use_llama_cpp)

    summaries = await generate_chunks(client, [t.text for t in texts])

    recap = await client.agenerate(
        prompt=prompt.format(texts="\n\n".join(summaries)),
        max_tokens=PARROT_CONFIGS.parrot_configs.generative_models.text_generation.max_tokens,
        temperature=PARROT_CONFIGS.parrot_configs.generative_models.text_generation.temperature,
    )

    return recap
