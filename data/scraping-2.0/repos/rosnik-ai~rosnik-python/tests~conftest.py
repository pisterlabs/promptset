import logging
import os
import sys

import pytest

from rosnik import config
from rosnik.events import queue


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}


@pytest.fixture
def openai():
    import openai

    yield openai

    # Reset import state ahead of the next test case
    mods = list(k for k in sys.modules.keys() if k.startswith("openai"))
    for m in mods:
        del sys.modules[m]


@pytest.fixture
def openai_client(openai):
    return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "test-key"))


@pytest.fixture
def openai_chat_completions_class(openai_client):
    cls = openai_client.chat.completions.__class__
    original_create = cls.create
    yield cls
    cls.create = original_create


@pytest.fixture
def openai_completions_class(openai_client):
    cls = openai_client.completions.__class__
    original_create = cls.create
    yield cls
    cls.create = original_create


@pytest.fixture(autouse=True)
def config_reset():
    yield
    config.Config = config._Config()


@pytest.fixture
def debug_logger(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture(autouse=True)
def event_queue(mocker):
    # Don't send process the event queue so we can inspect
    # it in tests.
    mocker.patch("rosnik.events.queue.EventProcessor")
    yield queue.event_queue
    # Clear queue
    while queue.event_queue.qsize() > 0:
        queue.event_queue.get(block=False)
    assert queue.event_queue.qsize() == 0
