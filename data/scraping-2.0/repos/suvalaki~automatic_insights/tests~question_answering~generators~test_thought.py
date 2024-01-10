import pytest
from typing import List

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import (
    Question,
    Hypothesis,
    Thought,
    TargettedThought,
    DataSourceSelection,
)
from ai.question_answering.generators.base import TargettedThoughtGenerator


class MockAlwaysFalseComparisonFilter:
    def __call__(self, hypothesis: Thought, other_hypothesis: List[Thought]) -> bool:
        return True


class MockStaticDataSourceSelector:
    def __call__(self, hypothesis: Hypothesis) -> Tool:
        return DataSourceSelection(
            data_source=Tool.from_function(
                func=lambda hypothesis: "a returned Hypothesis datum",
                name="test",
                description="test",
                # args_schema=[str],
            ),
            reason="test",
        )


class MockDiscussionGenerator:
    def __call__(self, hypothesis: Hypothesis, data: str) -> str:
        return "test discussion"


class MockDiscussionScorer:
    def __call__(
        self, hypothesis: Hypothesis, discussion: str, conclusion: str
    ) -> float:
        return 0.5


def test_static_thought_generator():
    hypothesis = Hypothesis(hypothesis="hypothesis", data_sources=[])

    generator = TargettedThoughtGenerator(
        MockAlwaysFalseComparisonFilter(),
        MockStaticDataSourceSelector(),
        MockDiscussionGenerator(),
        MockDiscussionScorer(),
    )

    reply = generator.generate(hypothesis, [])

    print(reply.json())


from ai.question_answering.data import LLMDataSourceSelector
from ai.question_answering.generators.thought.llm_discussion import (
    LLMHypothesisDataExplainer,
)
from ai.question_answering.generators.thought.llm_scoring import LLMDataExplainerScorer


def test_llm_generator():
    from langchain.chat_models import ChatOpenAI

    model_name = "gpt-4"
    temperature = 1.0
    model = ChatOpenAI(model_name=model_name, temperature=temperature)

    class ToolInput(BaseModel):
        question: str

    data_sources_desc = [
        "SQL table of sales data",
        "view of sales per country",
        "view of sales per product",
        "view of sales cadence per customer",
        "view of change in sales over time per customer",
    ]

    dummy_data = "Uk sales = 3%."

    data_sources = [
        Tool.from_function(
            func=lambda hypothesis: dummy_data,
            name=desc,
            description=desc,
            args_schema=ToolInput,
        )
        if desc == "view of sales per country"
        else Tool.from_function(
            func=lambda hypothesis: "",
            name=desc,
            description=desc,
            args_schema=ToolInput,
        )
        for desc in data_sources_desc
    ]

    generator = TargettedThoughtGenerator(
        MockAlwaysFalseComparisonFilter(),
        LLMDataSourceSelector(data_sources, model),
        LLMHypothesisDataExplainer(model),
        LLMDataExplainerScorer(model),
    )

    hypothesis = Hypothesis(
        hypothesis="The UK has the highest sales", data_sources=data_sources
    )

    thought = generator.generate(hypothesis, [])
    print()
    print(thought.data, thought.discussion, thought.score)
