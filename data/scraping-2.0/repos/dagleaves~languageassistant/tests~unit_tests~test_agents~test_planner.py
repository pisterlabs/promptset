import os

import pytest
from langchain.chat_models import ChatOpenAI

from languageassistant.agents.planner import load_lesson_planner
from languageassistant.agents.planner.schema import Lesson
from languageassistant.utils import load_openai_api_key

invalid_openai_api_key = os.getenv("OPENAI_API_KEY") in [None, "", "api_key"]


@pytest.mark.skipif(invalid_openai_api_key, reason="Needs valid OpenAI API key")
def setup_module() -> None:
    load_openai_api_key()


def test_schema_empty_lesson_repr() -> None:
    test_lesson = Lesson(topics=[])
    assert str(test_lesson) == ""


@pytest.mark.skipif(invalid_openai_api_key, reason="Needs valid OpenAI API key")
def test_schema_lesson_repr() -> None:
    test_lesson = Lesson(topics=["test"])
    assert str(test_lesson) == "1. test\n"


@pytest.mark.skipif(invalid_openai_api_key, reason="Needs valid OpenAI API key")
def test_initialize_planner() -> None:
    llm = ChatOpenAI(temperature=0)  # type: ignore[call-arg]
    load_lesson_planner(llm)


@pytest.mark.skipif(invalid_openai_api_key, reason="Needs valid OpenAI API key")
def test_planner_result() -> None:
    llm = ChatOpenAI(temperature=0)  # type: ignore[call-arg]
    agent = load_lesson_planner(llm)
    inputs = {
        "language": "Chinese",
        "proficiency": "Beginner",
    }
    agent.plan(inputs)
