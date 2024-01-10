import logging
import pickle
from copy import copy
from typing import Callable, Optional

import numpy as np
import openai
from langchain.embeddings.base import Embeddings

log = logging.getLogger(__name__)


class CachedOpenAIEmbeddings(Embeddings):
    def __init__(self, cache: Optional[dict[str, np.ndarray]] = None) -> None:
        super().__init__()
        # TODO: see todo in get_openai_embedding below
        self.get_openai_embedding = copy(get_openai_embedding)
        self.get_openai_embedding.cache = cache or {}

    def save_cache(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.cache, f)

    def load_cache(self, path: str):
        with open(path, "rb") as f:
            self.cache = pickle.load(f)

    @property
    def cache(self) -> dict:
        return self.get_openai_embedding.cache

    @cache.setter
    def cache(self, cache: dict[str, np.ndarray]):
        self.get_openai_embedding.cache = cache

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self.get_openai_embedding(text))


# Currently unused, copied over from a private project. Maybe useful later.
def cutoff_text_at_sentence(text: str, max_words=5000):
    """
    Cuts off text at sentence boundary if it exceeds max_words
    """
    words = text.split(" ")
    if len(words) < max_words:
        return text

    # get index of last word ending in "".""
    # The previous code ensures that `idx`` is always set, so we can safely ignore the
    # IDE warning below
    for idx, word in enumerate(words[max_words::-1]):
        if word.endswith("."):
            break
    last_word_index = max_words - idx + 1

    return " ".join(words[:last_word_index])


def _setup_cache_attribute(func: Callable) -> dict[str, np.ndarray]:
    """
    Sets up the cache attribute for the given function if it doesn't
    exist yet and returns its value.
    :param func:
    :return:
    """
    if not hasattr(func, "cache"):
        log.debug(f"No cache found for {func.__name__}. Creating one as empty dict.")
        func.cache = {}
    return func.cache


def _get_embedding_from_api(
    text: str, model="text-embedding-ada-002", timeout=100
) -> np.ndarray:
    response = openai.Embedding.create(input=[text], model=model, timeout=timeout)
    return np.array(response["data"][0]["embedding"])  # type: ignore


# TODO: copied over from a private project. Caching like this is probably a bad idea for our purposes, since
#  the .cache attribute for a function is global to the calling program. We should avoid re-computing
#  embeddings in a different, more explicit way. This is just a shortcut for now.
def get_openai_embedding(
    text: str, model="text-embedding-ada-002", timeout=100, use_cache=True
) -> np.ndarray:
    """
    Returns the embedding for the given text. Using explicit caching if desired to avoid unnecessary API calls.
    Access the cache via get_embedding.cache. We don't use functools b/c it doesn't give explicit access to the cache.

    :param text:
    :param model:
    :param timeout:
    :param use_cache: if True, the cache from `get_openai_embedding` is used. You can set and access at
        as `get_openai_embedding.cache`. If not set explicitly, uses an empty dict (so `use_cache=True` can still be used).
        The text is used as key in the cache.
    :return:
    """
    text = text.replace("\n", " ")

    if use_cache:
        cache = _setup_cache_attribute(get_openai_embedding)
        embedding = cache.get(text)
        if embedding is not None:
            log.debug("Found embedding in cache, skipping API call")
            return embedding
    embedding = _get_embedding_from_api(text, model=model, timeout=timeout)
    if use_cache:
        get_openai_embedding.cache[text] = embedding
    return embedding
