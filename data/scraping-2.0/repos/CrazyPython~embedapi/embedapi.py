import io
import json
import time
from typing import List

import requests
import tiktoken
import numpy as np
import more_itertools
from intermodel import callgpt

base_url = "http://localhost:7000/{}"


# todo: nptyping
def encode_one(transformer: str, datum: str) -> np.array:
    assert isinstance(datum, str)
    return encode_batch(transformer, [datum])[0]


encode_query = encode_one


def encode_batch(transformer: str, data: List[str]) -> np.array:
    if is_openai_model(transformer):
        return _openai_encode_batch(transformer, data)
    else:
        return _local_encode_batch(transformer, data)


encode_queries_batch = encode_batch


def encode_passages(transformer: str, data: List[str]) -> np.array:
    if is_openai_model(transformer):
        return _openai_encode_batch(transformer, data)
    else:
        if 'e5-' in transformer and not transformer.endswith(':symmetric'):
            for i in range(len(data)):
                data[i] = 'passage: ' + data[i]
        return _local_encode_batch(transformer, data)

def removesuffix(self: str, suffix: str, /) -> str:
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]

def _local_encode_batch(transformer: str, data: List[str]) -> np.array:
    transformer = removesuffix(transformer, ':symmetric')
    response = requests.post(base_url.format(transformer), json={"input": data})
    response.raise_for_status()
    virt_file = io.BytesIO(response.content)
    data = np.load(virt_file)
    return data


def _openai_encode_batch(transformer: str, data: List[str]) -> np.array:
    import openai

    assert transformer.startswith("text-embedding")
    encoding = tiktoken.encoding_for_model(transformer)
    result = []
    # possible alternative: vary by token count
    # https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
    for chunk in more_itertools.chunked(data, 148):  # 493 = e ** 6
        truncated_chunk = []
        for item in chunk:
            truncated_chunk.append(encoding.decode(encoding.encode(item)[:callgpt.max_token_length(transformer)]))
        for retry in range(6):
            try:
                response = openai.Embedding.create(model=transformer, input=truncated_chunk)
            except (json.decoder.JSONDecodeError, openai.error.APIConnectionError) as e:
                time.sleep(2 ** (retry - 1))
            except openai.error.RateLimitError as e:
                time.sleep(2 ** (retry - 1))
            else:
                result.extend(response["data"])
                break
        else:
            raise e
    return np.array([obj["embedding"] for obj in result], dtype=np.float32)


def is_openai_model(s):
    return (
        "ada" in s or "babbage" in s or "curie" in s or "cushman" in s or "davinci" in s
    )


# todo: add tests that check .shape
# ada is 1536
# all-mpnet-base-v2 is 768
