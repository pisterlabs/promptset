import pytest
from typing import List

from pydantic import BaseModel
from langchain.tools import Tool

from ai.question_answering.schema import Question, Hypothesis
from ai.question_answering.generators.base import HypothesisGenerator


class AlwaysFalseComparisonFilter:
    def __call__(
        self, hypothesis: Hypothesis, other_hypothesis: List[Hypothesis]
    ) -> bool:
        return True


class StaticMockHypothesisGenerator(HypothesisGenerator):
    def __init__(self):
        super().__init__(AlwaysFalseComparisonFilter())

    def _generate_single(self, question: Question) -> Hypothesis:
        return Hypothesis(hypothesis="hypothesis", data_sources=[])


def test_hypothesis_generator():
    generator = StaticMockHypothesisGenerator()
    assert True
    with pytest.raises(Exception):
        generator.generate(Question(question="question", data_sources=[]))


from ai.question_answering.generators.hypothesis.simple_llm import (
    LLMHypothesisGenerator,
)


def test_llm_hypothesis_generator():
    from langchain.chat_models import ChatOpenAI

    model_name = "gpt-3.5-turbo"
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
    data_sources = [
        Tool.from_function(
            func=lambda hypothesis: None,
            name=desc,
            description=desc,
            args_schema=ToolInput,
        )
        for desc in data_sources_desc
    ]

    generator = LLMHypothesisGenerator(
        comparison_filter=AlwaysFalseComparisonFilter(), llm=model
    )
    response = generator.generate(
        Question(
            question="Why are products in the UK doing so poorly?",
            data_sources=data_sources,
        ),
        [],
    )

    print(response.hypothesis)
