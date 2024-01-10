from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


P3_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading the anonimization.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P3_Q1_TEMPLATE = """
    Task: Based on the examples, give a score to the anonymized text, 0 is well anonimized and hard to re-identify, 1 is easy to re-identify or a large unique information leakage.
    Examples:
    TEXT: "{example_score_1}"
    SCORE: 1

    TEXT: "{example_score_0}"
    SCORE: 0

    TEXT: "{example_score_05}"
    SCORE: 0.5

    {format_instructions}

    Score the following text:
    "{user_input}"
    """


class P3Q1Parser(BaseModel):
    score: float = Field(
        description="anonimization score between 0-1, where 1 is very easy to re-identify and 0 is very hard to re-identify"
    )


process_3_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P3_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P3_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P3Q1Parser)),
    ],
)
