import pytest
from langchain.globals import set_debug

from weather_chat_ai.base import WeatherChatAI


@pytest.fixture
def chain():
    set_debug(False)
    chain = WeatherChatAI()
    assert chain is not None, "chain should have been created"
    return chain


@pytest.fixture
def inputs():
    return [
        [
            "What is the weather today in Breckenridge?",
            "answer should mention location",
        ],
        ["what about tomorrow?", "answer2 should have remembered location"],
    ]


def assert_answer(answer, location_assertion):
    # print(answer)
    assert isinstance(answer, str), "answer should be a string"
    assert len(answer) > 0, "answer should not be empty"
    assert "breckenridge" in answer.lower(), location_assertion


@pytest.mark.focus
def test_invoke_with_memory(chain, inputs):
    for input, location_assertion in inputs:
        answer = chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": "session_id"}},
        ).content
        assert_answer(answer, location_assertion)


@pytest.mark.asyncio
# @pytest.mark.focus
async def test_ainvoke_with_memory(chain, inputs):
    for input, location_assertion in inputs:
        result = await chain.ainvoke(
            {"input": input},
            config={"configurable": {"session_id": "session_id"}},
        )
        answer = result.content
        assert_answer(answer, location_assertion)


# @pytest.mark.focus
def test_stream_with_memory(chain, inputs):
    for input, location_assertion in inputs:
        answer = ""
        for chunk in chain.stream(
            {"input": input},
        ):
            answer += chunk.content

        assert_answer(answer, location_assertion)


@pytest.mark.asyncio
# @pytest.mark.focus
async def test_astream_with_memory(chain, inputs):
    for input, location_assertion in inputs:
        answer = ""
        async for chunk in chain.astream(
            {"input": input},
        ):
            answer += chunk.content

        assert_answer(answer, location_assertion)
