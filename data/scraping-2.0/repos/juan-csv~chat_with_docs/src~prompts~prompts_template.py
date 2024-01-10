""" This module contains the prompts templates for the different chains. """

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from pydantic import Field

INIT_QUERY = """Create top 3 modifications for improving document and be more secure for the client. 
Provide the sugestion of modifications including the paragraph to modify and the new paragraph modified.
Explain why it's necessary the modification."""


# ---------------------------------------------------------
# ConversationalRetrievalChain
# ---------------------------------------------------------
prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Chat History: {chat_history}
Answer:
"""
CONVERSATIONAL_RETRIEVAL_CHAIN_V2 = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "chat_history"]
)


# CONVERSATIONAL_RETRIEVAL_CHAIN_V2 = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Chat History: {chat_history}
# Answer:
# """
# ---------------------------------------------------------
# Text Analysis Chains
# ---------------------------------------------------------

summarize_prompt_template = """Summarize the following text. Keep the original language in 
which the text is written. The summary has to be shorter than the original text. Don't add up or make up
any information not present in the original text.\nText: {text}"""
SUMMARIZE_CHAIN = PromptTemplate.from_template(summarize_prompt_template)

change_of_tone_prompt_template = "Rewrite the following text to match a {tone_description} tone.\nText: {text}"
CHANGE_TONE_CHAIN = PromptTemplate.from_template(
    change_of_tone_prompt_template)

rephrase_prompt_template = "For the following text, rephrase it.\nText: {text}"
REPHRASE_CHAIN = PromptTemplate.from_template(rephrase_prompt_template)


# ---------------------------------------------------------
# Parragraph Suggestions
# ---------------------------------------------------------
TEMPLATE_PARRAGRAPH_SUGGESTION = """Use the following pieces to generate one suggestion to modify clauses in the contract provided below that may potentially harm the client's interests. Highlight potential risks and propose alternative language or conditions to mitigate these risks.

{context}

The output should be formatted as a the schema below:

Original parragraph: this is the original parragraph that we want to modify
Modified parragraph: this is the modified parragraph
Explanation: this is the explanation of why the parragraph should be modified

Do not add any other text to the output.
Do not add numbers or letters or make any modification to the keys "Original parragraph", "Modified parragraph" and "Explanation".

"""

PROMPT_PARRAGRAPH_SUGGESTION = PromptTemplate(
    template=TEMPLATE_PARRAGRAPH_SUGGESTION,
    input_variables=["context"],
)


# ---------------------------------------------------------
# MapReduce Suggestions
# ---------------------------------------------------------
class MapOutput(PydanticOutputParser):
    """Map output"""
    original_parragraph: list = Field(
        description="List of three original parragraphs")
    modified_parragraph: list = Field(
        description="List of three modified parragraphs")
    explanation: list = Field(
        description="List of three explanations of why the parragraph should be modified")


parser_reduce = PydanticOutputParser(pydantic_object=MapOutput)

PROMPT_DOC = PromptTemplate(
    template="{page_content}",
    input_variables=["page_content"]
)

TEMPLATE_MAP_SUGGESTION_MAP = """Use the following pieces to generate one suggestion to modify clauses in the contract provided below that may potentially harm the client's interests. Highlight potential risks and propose alternative language or conditions to mitigate these risks.

{context}

The output should be formatted as a the schema below:

Original parragraph: this is the original parragraph that we want to modify
Modified parragraph: this is the modified parragraph
Explanation: this is the explanation of why the parragraph should be modified

Do this for the top 3 suggestions.
Do not add any other text to the output.
Do not add numbers or letters or make any modification to the keys "Original parragraph", "Modified parragraph" and "Explanation".

"""

PROMPT_MAP_SUGGESTION = PromptTemplate(
    template=TEMPLATE_MAP_SUGGESTION_MAP,
    input_variables=["context"],
    # partial_variables={
    #     "format_instructions": parser_map.get_format_instructions()},
    # output_parser=parser_map
)

TEMPLATE_REDUCE_SUGGESTION = """Select from the following suggestions the best three suggestions.

{context}

The output should be formatted as a the schema below:

Original parragraph: this is the original parragraph that we want to modify
Modified parragraph: this is the modified parragraph
Explanation: this is the explanation of why the parragraph should be modified

Do this for the top 3 suggestions.
Do not add any other text to the output.
Do not add numbers or letters or make any modification to the keys "Original parragraph", "Modified parragraph" and "Explanation".

{format_instructions}

"""
PROMPT_REDUCE_SUGGESTION = PromptTemplate(
    template=TEMPLATE_REDUCE_SUGGESTION,
    input_variables=["context"],
    partial_variables={
        "format_instructions": parser_reduce.get_format_instructions()},
)

# ---------------------------------------------------------
# Other prompt
# ---------------------------------------------------------
