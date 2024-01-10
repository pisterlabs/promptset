# https://github.com/hwchase17/langchain/blob/master/libs/langchain/langchain/agents/agent_toolkits/sql/base.py
# https://github.com/hwchase17/langchain/blob/master/libs/langchain/langchain/agents/mrkl/base.py


# Define the geneative capabilities of the Hypothesis Answer Generator
# as tools

from typing import Callable, List

from langchain.tools import Tool
from langchain.agents.agent_toolkits.base import BaseToolkit

from ai.question_answering.generators.base import (
    Thought,
    TargettedThoughtGenerator,
    ThoughtSummarizer,
    Hypothesis,
    HypothesisAnswerGenerator,
    HypothesisConcluder,
)


def create_toolkit(
    hypothesis: Hypothesis,
    thoughts: List[Thought],  # The weakref to thoughts
    thought_generator: TargettedThoughtGenerator,
    thought_summarizer: ThoughtSummarizer,
    concluder: HypothesisConcluder,
    evaluator: Callable[[Hypothesis, List[Thought], Thought], bool],
):
    thought_generator_tool = Tool.from_function(
        func=thought_generator.generate(hypothesis, thoughts),
        name="Evidence Thought Getter",
        description="Queries a data getter to provide evidence and discusses that data.",
    )

    thought_summarizer = Tool.from_function(
        func=
    )
