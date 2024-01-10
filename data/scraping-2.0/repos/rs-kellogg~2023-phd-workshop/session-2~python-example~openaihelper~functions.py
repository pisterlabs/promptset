from pathlib import Path
from tqdm import tqdm
import pandas as pd
import openai
import yaml
import tiktoken
from typing import Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def config(config_file: Path) -> Dict:
    with open(config_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        return conf


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs) -> str:
    return openai.ChatCompletion.create(**kwargs)


def chat_complete(text: str, model_name: str, prompt: str) -> str:
    assert text is not None and len(text) > 0
    assert model_name is not None and len(model_name) > 0
    assert prompt is not None and len(prompt) > 0

    try:
        return completion_with_backoff(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt + " " + text},
            ],
        )
    except openai.InvalidRequestError as e:
        return str(e)
    except openai.error.RateLimitError as e:
        return str(e)
    except Exception as e:
        return str(e)


def count_tokens(text: str, encoding_name: str) -> int:
    assert text is not None and len(text) > 0
    assert encoding_name is not None and len(encoding_name) > 0

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def validate_result(result: str) -> bool:
    assert result is not None and len(result) > 0
    return not result.startswith("InvalidRequestError") and not result.startswith("RateLimitError")