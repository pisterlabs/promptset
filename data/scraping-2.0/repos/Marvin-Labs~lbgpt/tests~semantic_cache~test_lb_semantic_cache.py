# -*- coding: utf-8 -*-
import asyncio
import os
import random
import string
import tempfile
import time

import pytest
from langchain.embeddings import HuggingFaceEmbeddings
from pytest_mock import MockerFixture

from lbgpt import ChatGPT
from lbgpt.semantic_cache import FaissSemanticCache, QdrantSemanticCache
from lbgpt.types import ChatCompletionAddition

# for qdrant we want to get truly random names. However, it may be that the random seed is set somewhere else,
# so we have to create an instance of random.Random with a new seed here.
rng = random.Random(time.time())


TEST_EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="bert-base-uncased")

SEMANTIC_CACHES = [
    lambda: FaissSemanticCache(
        embedding_model=TEST_EMBEDDING_MODEL,
        cosine_similarity_threshold=0.95,
        path=tempfile.TemporaryDirectory().name,
    ),
    lambda: QdrantSemanticCache(
        embedding_model=TEST_EMBEDDING_MODEL,
        cosine_similarity_threshold=0.95,
        host="localhost",
        port=6333,
        collection_name="".join(rng.choice(string.ascii_letters) for _ in range(20)),
    ),
]


def get_single_request_content_for_test(user_message: str):
    messages = [
        {"role": "user", "content": user_message},
    ]
    return dict(
        messages=messages,
        model="gpt-3.5-turbo-0613",
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        request_timeout=10,
    )


@pytest.mark.parametrize("semantic_cache", SEMANTIC_CACHES)
@pytest.mark.vcr
def test_chatgpt_cache_exact(mocker: MockerFixture, semantic_cache):
    semantic_cache = semantic_cache()
    single_request_content = get_single_request_content_for_test(
        "please respond with pong"
    )

    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
        semantic_cache=semantic_cache,
    )

    # Setting the cache
    cache_interaction = mocker.spy(semantic_cache, "query_cache")

    # run with an empty cache
    res = asyncio.run(
        lb.chat_completion_list([single_request_content], show_progress=False)
    )

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": hash(semantic_cache)}

    if hasattr(semantic_cache, "count"):
        cache_stats["count"] = semantic_cache.count

    # Getting from cache
    res_cache = asyncio.run(
        lb.chat_completion_list([single_request_content], show_progress=False)
    )

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    if hasattr(semantic_cache, "count"):
        assert semantic_cache.count == cache_stats["count"]
    assert hash(semantic_cache) == cache_stats["hash"]

    assert res_cache[0].is_exact is True


@pytest.mark.parametrize("semantic_cache", SEMANTIC_CACHES)
@pytest.mark.vcr
def test_chatgpt_cache_inexact(mocker: MockerFixture, semantic_cache):
    semantic_cache = semantic_cache()

    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
        semantic_cache=semantic_cache,
    )

    # Setting the cache
    cache_interaction = mocker.spy(semantic_cache, "query_cache")

    # run with an empty cache
    res = asyncio.run(
        lb.chat_completion_list(
            [get_single_request_content_for_test("please respond with pong")],
            show_progress=False,
        )
    )

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": hash(semantic_cache)}

    if hasattr(semantic_cache, "count"):
        cache_stats["count"] = semantic_cache.count

    # Getting from cache
    res_cache = asyncio.run(
        lb.chat_completion_list(
            [get_single_request_content_for_test("please respond with pongo")],
            show_progress=False,
        )
    )

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert isinstance(cache_interaction.spy_return, ChatCompletionAddition)

    # asserting that no items were added to the cache
    if hasattr(semantic_cache, "count"):
        assert semantic_cache.count == cache_stats["count"]
    assert hash(semantic_cache) == cache_stats["hash"]

    assert res_cache[0].is_exact is False


@pytest.mark.parametrize("semantic_cache", SEMANTIC_CACHES)
@pytest.mark.vcr
def test_chatgpt_cache_failed(mocker: MockerFixture, semantic_cache):
    semantic_cache = semantic_cache()

    lb = ChatGPT(
        api_key=os.environ["OPEN_AI_API_KEY"],
        stop_after_attempts=1,
        stop_on_exception=True,
        semantic_cache=semantic_cache,
    )

    # Setting the cache
    cache_interaction = mocker.spy(semantic_cache, "query_cache")

    # run with an empty cache
    res = asyncio.run(
        lb.chat_completion_list(
            [get_single_request_content_for_test("please respond with pong")],
            show_progress=False,
        )
    )

    # asserting that the cache was not called
    assert cache_interaction.call_count == 1
    assert cache_interaction.spy_return is None

    # some cache stats and hash
    cache_stats = {"hash": hash(semantic_cache)}

    if hasattr(semantic_cache, "count"):
        cache_stats["count"] = semantic_cache.count

    # Getting from cache
    res_cache = asyncio.run(
        lb.chat_completion_list(
            [get_single_request_content_for_test("please respond with test")],
            show_progress=False,
        )
    )

    # asserting that the cache was called and returned the values
    assert cache_interaction.call_count == 2
    assert cache_interaction.spy_return is None

    # asserting that no items were added to the cache
    if hasattr(semantic_cache, "count"):
        assert semantic_cache.count == cache_stats["count"] + 1
    assert hash(semantic_cache) == cache_stats["hash"]

    assert res_cache[0].is_exact is True
