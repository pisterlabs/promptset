import uuid

import pytest
from openai._base_client import APITimeoutError

from greptimeai import collector
from . import sync_client
from ..database.db import truncate_tables, get_trace_data_with_retry


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat_completion_retry(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"
    try:
        sync_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "1+1=",
                }
            ],
            model=model,
            user=user_id,
            seed=1,
            timeout=0.1,
        )
    except Exception as e:
        assert isinstance(e, APITimeoutError)

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id=user_id, retry=3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "openai_completion" == trace.get("span_name")
    assert "openai" == trace.get("span_attributes", {}).get("source")

    assert {"client.chat.completions.create", "retry", "exception", "end"} == {
        event.get("name") for event in trace.get("span_events", [])
    }
