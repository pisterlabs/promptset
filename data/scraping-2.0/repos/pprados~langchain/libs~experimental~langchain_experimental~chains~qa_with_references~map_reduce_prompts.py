# flake8: noqa

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser
from .references import references_parser, References

_map_verbatim_parser = LineListOutputParser()


# Use some object. It's easier to update the verbatims schema.
_response_example_1 = References(
    response="This Agreement is governed by English law.",
    documents_ids=["_idx_0", "_idx_1"],
)

_response_example_2 = References(response="", documents_ids=[])

_question_prompt_template = """
Use the following portion of a long document to see if any of the text is relevant to answer the question.
---
{context}
---

Question: {question}

Extract all verbatims from texts relevant to answering the question in separate strings else output an empty array.
{format_instructions}

"""

QUESTION_PROMPT = PromptTemplate(
    template=_question_prompt_template,
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": _map_verbatim_parser.get_format_instructions()
    },
    output_parser=_map_verbatim_parser,
)

_combine_prompt_template = """Given the following extracts from several documents, 
a question and not prior knowledge. 
QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law
ids: _idx_0,_idx_1
Content: 
ids: _idx_2
Content: 
ids: _idx_3
Content: The english law is applicable for this agreement.
ids:_idx_4
=========
FINAL ANSWER:
{response_example_1}

QUESTION: What did the president say about Michael Jackson?
=========
Content: "Madam Speaker, Madam Vice President, our First Lady and Second "
"Gentleman. Members of Congress and the Cabinet."
ids: _idx_0
Content: 
ids: _idx_1,_idx_2
Content: 
ids: _idx_3
Content: 
ids: _idx_4
=========
FINAL ANSWER: 
{response_example_2}

QUESTION: {question}
=========
{summaries}
=========
If you are not confident with your answer, say 'I don't know'. 
{format_instructions}
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=_combine_prompt_template,
    input_variables=["summaries", "question"],
    partial_variables={
        "format_instructions": references_parser.get_format_instructions(),
        "response_example_1": str(_response_example_1),
        "response_example_2": str(_response_example_2),
    },
    output_parser=references_parser,
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\n" "Ids: {_idx}",
    input_variables=["page_content", "_idx"],
)
