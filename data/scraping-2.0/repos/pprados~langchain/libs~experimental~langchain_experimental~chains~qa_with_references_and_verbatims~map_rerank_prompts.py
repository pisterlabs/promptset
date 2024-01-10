# flake8: noqa
from langchain.output_parsers import PydanticOutputParser, RegexParser
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

from .verbatims import VerbatimsFromDoc, Verbatims

_map_verbatim_parser: BaseOutputParser = PydanticOutputParser(
    pydantic_object=VerbatimsFromDoc
)

_rank_parser = RegexParser(
    regex=r"(.*)\n?Score: (\d*)",
    output_keys=["answer", "score"],
)

_response_example_1 = Verbatims(
    response="red", documents=[VerbatimsFromDoc(ids=[99], verbatims=["Apples are red"])]
)
_response_example_2 = Verbatims(
    response="a sports car or an suv",
    documents=[
        VerbatimsFromDoc(
            ids=[99], verbatims=["he was not sure if it was a sports car or an suv"]
        )
    ],
)
_response_example_3 = Verbatims(
    response="This document does not answer the question", documents=[]
)
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
- extract all the verbatims from the texts only if they are relevant to answering the question, in a list of strings 
- answers the question
- calculates a score of how fully it answered the user's question
- creates a final answer

The ids must be only in the form '_idx_<number>'.
{format_instructions}

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
Helpful Answer: {response_example_1}
Score: 100

Example #2
Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: {response_example_2}
Score: 60

Example #3
Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: {response_example_3}
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
    partial_variables={
        "format_instructions": _map_verbatim_parser.get_format_instructions(),
        "response_example_1": _response_example_1.json(),
        "response_example_2": _response_example_2.json(),
        "response_example_3": _response_example_3.json(),
    },
    output_parser=_rank_parser,
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nIds: {_idx}",
    input_variables=["page_content", "_idx"],
)
