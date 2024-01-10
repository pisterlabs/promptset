import httpx

from calliope.inference.engines.hugging_face import text_to_text_inference_hugging_face
from calliope.inference.engines.openai_text import openai_text_to_text_inference
from calliope.inference.engines.replicate import replicate_text_to_text_inference
from calliope.models import (
    InferenceModelProvider,
    KeysModel,
)
from calliope.tables import ModelConfig


async def text_to_text_inference(
    httpx_client: httpx.AsyncClient,
    text: str,
    model_config: ModelConfig,
    keys: KeysModel,
) -> str:
    """
    Performs a text->text inference using an LLM.

    Args:
        httpx_client: the async HTTP session.
        text: the input text, to be sent as a prompt.
        model_config: the ModelConfig with model and parameters.
        keys: API keys, etc.

    Returns:
        the generated text.
    """
    print(f"text_to_text_inference: {model_config.slug}, {model_config.model}")
    model = model_config.model

    if model.provider == InferenceModelProvider.HUGGINGFACE:
        print(f"text_to_text_inference.huggingface {model.provider_model_name}")
        extended_text = await text_to_text_inference_hugging_face(
            httpx_client, text, model_config, keys
        )
        print(f'extended_text="{extended_text}"')
    elif model.provider == InferenceModelProvider.OPENAI:
        print(f"text_to_text_inference.openai {model.provider_model_name}")
        extended_text = await openai_text_to_text_inference(
            httpx_client, text, model_config, keys
        )
        print(f'extended_text="{extended_text}"')
    elif model.provider == InferenceModelProvider.REPLICATE:
        print(f"text_to_text_inference.replicate {model.provider_model_name}")
        extended_text = await replicate_text_to_text_inference(
            httpx_client, text, model_config, keys
        )
        print(f'extended_text="{extended_text}"')
    else:
        raise ValueError(
            "Don't know how to do text->text inference for provider "
            f"{model.provider}."
        )

    return extended_text
