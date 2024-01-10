# flake8: noqa
from typing import Type

from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from .references import References

_rank_parser = RegexParser(
    regex=r"(.*)\n?Score: (\d*)",
    output_keys=["answer", "score"],
)


class ReferenceOutputParser(BaseOutputParser[References]):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[References] = References
    """The pydantic model to parse."""

    def parse(self, text: str) -> References:
        return References(response=text)

    def get_format_instructions(self) -> str:
        return ""

    @property
    def _type(self) -> str:
        return "reference"


rerank_reference_parser = ReferenceOutputParser()

prompt_template = """
Given the following extracts from several documents, a question and not prior knowledge. 

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

The ids must be only in the form '_idx_<number>'.

Process step by step:
- extract the references ("IDS")
- answers the question
- calculates a score of how fully it answered the user's question
- creates a final answer

The ids must be only in the form '_idx_<number>'.
This should be in the following format:
Question: [question here]
Helpful Answer: [json answer here]
Score: [to the next line, score between 0 and 100]

Example #1
Context:
---------
Apples are red. The car is blue.
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2
Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3
Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=_rank_parser,
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nIds: {_idx}\n",
    input_variables=["page_content", "_idx"],
)
