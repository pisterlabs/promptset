import http.client
import io
import json
import os
import random
import re
import uuid

import openai
import pinecone
import pytest
import requests
import responses
from urllib3.response import HTTPResponse

import gentrace

PINECONE_API_PATTERN = re.compile("^https?:\/\/.*pinecone\.io\/.*")


def test_pinecone_pipeline_fetch_server(setup_teardown_pinecone):
    responses.add_passthru(PINECONE_API_PATTERN)

    pipeline = gentrace.Pipeline(
        "test-gentrace-python-pipeline",
        host="http://localhost:3000/api",
        pinecone_config={
            "api_key": os.getenv("PINECONE_API_KEY"),
        },
    )

    pipeline.setup()

    runner = pipeline.start()

    pinecone = runner.get_pinecone()

    index = pinecone.Index("openai-trec")

    index.fetch(ids=["3980"])

    info = runner.submit()

    assert uuid.UUID(info["pipelineRunId"]) is not None


def test_pinecone_pipeline_query_server(vector, setup_teardown_pinecone):
    responses.add_passthru(PINECONE_API_PATTERN)

    pipeline = gentrace.Pipeline(
        "test-gentrace-python-pipeline",
        host="http://localhost:3000/api",
        pinecone_config={
            "api_key": os.getenv("PINECONE_API_KEY"),
        },
    )

    pipeline.setup()

    runner = pipeline.start()

    pinecone = runner.get_pinecone()

    index = pinecone.Index("openai-trec")

    index.query(top_k=10, vector=vector, pipline_id="self-contained-pinecone-query")

    info = runner.submit()

    assert uuid.UUID(info["pipelineRunId"]) is not None


def test_pinecone_pipeline_list_indices_server(setup_teardown_pinecone):
    responses.add_passthru(PINECONE_API_PATTERN)

    pipeline = gentrace.Pipeline(
        "test-gentrace-python-pipeline",
        host="http://localhost:3000/api",
        pinecone_config={
            "api_key": os.getenv("PINECONE_API_KEY"),
        },
    )

    pipeline.setup()

    runner = pipeline.start()

    pinecone = runner.get_pinecone()

    active_indexes = pinecone.list_indexes()

    # liste_indexes is not supported, do not send information
    info = runner.submit()

    assert info["pipelineRunId"] is None


def test_pinecone_pipeline_upsert_server(vector, setup_teardown_pinecone):
    responses.add_passthru(PINECONE_API_PATTERN)

    pipeline = gentrace.Pipeline(
        "test-gentrace-python-pipeline",
        host="http://localhost:3000/api",
        pinecone_config={
            "api_key": os.getenv("PINECONE_API_KEY"),
        },
    )

    pipeline.setup()

    runner = pipeline.start()

    pinecone = runner.get_pinecone()

    pinecone.Index("openai-trec").upsert(
        [
            {
                "id": str(random.randint(0, 9999)),
                "values": vector,
            },
        ],
        pipeline_id="self-contained-pinecone-upsert",
    )

    # list_indexes is not supported, do not send information
    info = runner.submit()

    assert uuid.UUID(info["pipelineRunId"]) is not None


def test_pinecone_self_contained_fetch_server(setup_teardown_pinecone):
    responses.add_passthru(PINECONE_API_PATTERN)

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
    )

    result = pinecone.Index("openai-trec").fetch(
        ids=["3980"], pipeline_id="self-contained-pinecone-fetch"
    )

    assert uuid.UUID(result["pipelineRunId"]) is not None
    print(setup_teardown_pinecone)


def test_pinecone_self_contained_fetch_server_with_slug(setup_teardown_pinecone):
    responses.add_passthru(PINECONE_API_PATTERN)

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
    )

    result = pinecone.Index("openai-trec").fetch(
        ids=["3980"], pipeline_slug="self-contained-pinecone-fetch"
    )

    assert uuid.UUID(result["pipelineRunId"]) is not None
    print(setup_teardown_pinecone)


def test_pinecone_self_contained_query_server(setup_teardown_pinecone, vector):
    responses.add_passthru(PINECONE_API_PATTERN)

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
    )

    index = pinecone.Index("openai-trec")
    result = index.query(
        top_k=10, vector=vector, pipeline_id="self-contained-pinecone-query"
    )

    assert uuid.UUID(result["pipelineRunId"]) is not None
    print(setup_teardown_pinecone)


def test_pinecone_self_contained_query_server_no_pipeline_id(
    setup_teardown_pinecone, vector
):
    responses.add_passthru(PINECONE_API_PATTERN)

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
    )

    index = pinecone.Index("openai-trec")
    result = index.query(top_k=10, vector=vector)

    assert "pipelineRunId" not in result
    print(setup_teardown_pinecone)
