from typing import List, Optional, Union

import openai
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sum_pai.constants import EMBEDDING_MODEL


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(4),
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
)
def get_embedding(
    text_or_tokens: Union[str, List[str]],
    model: str = EMBEDDING_MODEL,
    hashed_key: Optional[str] = None,
) -> List[float]:
    """Generates an embedding for the given text or tokens using OpenAI's API.

    Args:
        text_or_tokens (Union[str, List[str]]): The text or tokens to generate the
          embedding for.
        model (str, optional): The name of the embedding model.
          Defaults to EMBEDDING_MODEL.
        hashed_key (Optional[str], optional): An optional hashed key for caching.
          Defaults to None.

    Returns:
        List[float]: The generated embedding.
    """
    result = openai.Embedding.create(input=text_or_tokens, model=model)
    return result["data"][0]["embedding"]  # type: ignore
