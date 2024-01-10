import uuid

import pytest
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from greptimeai import collector
from greptimeai.langchain.callback import GreptimeCallbackHandler

from ..database.db import get_trace_data_with_retry, truncate_tables


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"

    callback = GreptimeCallbackHandler()
    chat = ChatOpenAI(model=model)
    prompt = PromptTemplate.from_template("1 + {number} = ")

    chain = LLMChain(llm=chat, prompt=prompt, callbacks=[callback])
    result = chain.run(number=1, callbacks=[callback], metadata={"user_id": user_id})
    assert "2" in result

    collector.otel._force_flush()

    trace = get_trace_data_with_retry(user_id, "langchain_llm", 3)

    assert trace is not None

    assert "greptimeai" == trace.get("resource_attributes", {}).get("service.name")
    assert "langchain" == trace.get("span_attributes", {}).get("source")

    assert ["llm_start", "llm_end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert trace.get("model", "").startswith(model)
    assert trace.get("prompt_tokens", 0) > 10
    assert trace.get("completion_tokens", 0) >= 1
