import logging
import os
import time
from datetime import datetime

from llm_bench_api.config import CloudConfig
from openai import OpenAI


logger = logging.getLogger(__name__)


NON_CHAT_MODELS = []


def process_non_chat_model(client, config, run_config):
    raise NotImplementedError
    # return (
    #     client.completions.create(
    #         model=config.model_name,
    #         prompt=run_config["query"],
    #         max_tokens=run_config["max_tokens"],
    #         stream=True,
    #     ),
    #     "text",
    # )


def process_chat_model(client, config, run_config):
    return (
        client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": run_config["query"]}],
            max_tokens=run_config["max_tokens"],
            stream=True,
            extra_headers={
                "HTTP-Referer": "llm-benchmarks.com",
                "X-Title": "LLM Benchmarks",
            },
        ),
        "choices",
    )


def generate(config: CloudConfig, run_config: dict) -> dict:
    """Run OpenRouter inference and return metrics."""

    assert config.provider == "openrouter", "provider must be 'openrouter'"
    assert "query" in run_config, "query must be in run_config"
    assert "max_tokens" in run_config, "max_tokens must be in run_config"

    # Set up connection
    client = OpenAI(
        base_url=os.environ["OPENROUTER_BASE_URL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Generate
    time_0 = time.time()
    first_token_received = False
    previous_token_time = None
    output_chunks = 0
    output_tokens = 0
    times_between_tokens = []
    time_to_first_token = 0
    response_str = ""

    process_func = process_non_chat_model if config.model_name in NON_CHAT_MODELS else process_chat_model
    stream, response_key = process_func(client, config, run_config)

    for chunk in stream:
        if config.model_name in NON_CHAT_MODELS:
            response = chunk.choices[0]
            response_content = getattr(response, response_key)
        else:
            response = chunk.choices[0].delta  # type: ignore
            response_content = response.content if response is not None else None

        if response_content is not None:
            current_time = time.time()
            if not first_token_received:
                time_to_first_token = current_time - time_0
                first_token_received = True
            else:
                assert previous_token_time is not None
                times_between_tokens.append(current_time - previous_token_time)
            previous_token_time = current_time
            response_str += response_content
            output_chunks += 1
            if len(chunk.choices) == 1:
                output_tokens += 1
            else:
                raise ValueError("Unexpected number of choices")

    time_1 = time.time()
    generate_time = time_1 - time_0
    tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0

    metrics = {
        "gen_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_tokens": run_config["max_tokens"],
        "output_tokens": output_tokens,
        "generate_time": generate_time,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token": time_to_first_token,
        "times_between_tokens": times_between_tokens,
    }

    return metrics
