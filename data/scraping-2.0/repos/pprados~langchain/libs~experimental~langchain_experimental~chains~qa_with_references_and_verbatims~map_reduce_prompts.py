# flake8: noqa

from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser

from .verbatims import VerbatimsFromDoc, verbatims_parser, Verbatims

_map_verbatim_parser = LineListOutputParser()

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

# _map_verbatim_parser: BaseOutputParser = PydanticOutputParser(
#     pydantic_object=VerbatimsFromDoc
# )

# Use some object. It's easier to update the verbatims schema.
_response_example_1 = Verbatims(
    response="This Agreement is governed by English law.",
    documents=[
        VerbatimsFromDoc(
            # ids=["_idx_0"],
            ids=[""],
            verbatims=[
                "This Agreement is governed by English law",
                # "The english law is applicable for this agreement."
            ],
        ),
        VerbatimsFromDoc(
            # ids=["_idx_4"],
            ids=[""],
            verbatims=["The english law is applicable for this agreement."],
        ),
    ],
)

_response_example_2 = Verbatims(response="", documents=[])

_sample_1 = VerbatimsFromDoc(
    # ids=["_idx_0", "_idx_1"],
    ids=[],
    verbatims=["This Agreement is governed by English law"],
)
_sample_2 = VerbatimsFromDoc(
    # ids=["_idx_2"],
    ids=[],
    verbatims=[],
)
_sample_3 = VerbatimsFromDoc(
    # ids=["_idx_3"],
    ids=[],
    verbatims=[],
)
_sample_4 = VerbatimsFromDoc(
    # ids=["_idx_4"],
    ids=[],
    verbatims=["The english law is applicable for this agreement."],
)

_sample_5 = VerbatimsFromDoc(
    # ids=["_idx_0"],
    ids=[],
    verbatims=[
        "Madam Speaker, Madam Vice President, our First Lady and Second "
        "Gentleman. Members of Congress and the Cabinet."
    ],
)
_sample_6 = VerbatimsFromDoc(
    # ids=["_idx_1", "_idx_2"],
    ids=[],
    verbatims=[],
)
_sample_7 = VerbatimsFromDoc(
    # ids=["_idx_3"],
    ids=[],
    verbatims=[],
)
_sample_8 = VerbatimsFromDoc(
    # ids=["_idx_4"],
    ids=[],
    verbatims=[],
)

# _question_prompt_template = """
# Use the following portion of a long document to see if any of the text is relevant to answer the question.
# ---
# {context}
# ---
#
# Question: {question}
#
# Extract all verbatims from texts relevant to answering the question in separate strings else output an empty array.
# {format_instructions}
#
# """
#
# QUESTION_PROMPT = PromptTemplate(
#     template=_question_prompt_template,
#     input_variables=["context", "question"],
#     partial_variables={
#         "format_instructions": _map_verbatim_parser.get_format_instructions()
#     },
#     output_parser=_map_verbatim_parser,
# )

_combine_prompt_template = """Given the following extracts from several documents, 
a question and not prior knowledge. 

Process step by step:
- extract all verbatims
- extract all associated ids
- create a final response with these verbatims
- produces the json result

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: {_sample_1}
Content: {_sample_2}
Content: {_sample_3}
Content: {_sample_4}
=========
FINAL ANSWER:
{response_example_1}

QUESTION: What did the president say about Michael Jackson?
=========
Content: {_sample_5}
Content: {_sample_6}
Content: {_sample_7}
Content: {_sample_8}
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
        "format_instructions": verbatims_parser.get_format_instructions(),
        "response_example_1": _response_example_1.json(),
        "response_example_2": _response_example_2.json(),
        "_sample_1": _sample_1.json(),
        "_sample_2": _sample_2.json(),
        "_sample_3": _sample_3.json(),
        "_sample_4": _sample_4.json(),
        "_sample_5": _sample_5.json(),
        "_sample_6": _sample_6.json(),
        "_sample_7": _sample_7.json(),
        "_sample_8": _sample_8.json(),
    },
    output_parser=verbatims_parser,
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Ids: {_idx}\n" "Content: {page_content}\n",
    input_variables=["page_content", "_idx"],
)
