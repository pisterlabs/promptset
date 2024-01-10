from langchain.prompts import PromptTemplate
CONCISE_SUMMARY_MAP_PROMPT_TEMPLATE = """Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""

CONCISE_SUMMARY_MAP_PROMPT = PromptTemplate(
    template=CONCISE_SUMMARY_MAP_PROMPT_TEMPLATE, 
    input_variables=["text"]
)  

CONCISE_SUMMARY_COMBINE_PROMPT_TEMPLATE = """Write a concise, comprehensive summary of the following:


"{text}"

Also provide up to five suggested follow-up questions as a bulleted list. Only include questions that are 
likely answerable by the text and are not already answered in the summary you provided.

CONCISE SUMMARY:"""

CONCISE_SUMMARY_COMBINE_PROMPT = PromptTemplate(
    template=CONCISE_SUMMARY_COMBINE_PROMPT_TEMPLATE, 
    input_variables=["text"]
)  


CONCISE_SUMMARY_PROMPT_TEMPLATE = """Write a concise, comprehensive summary of the following:


"{text}"


CONCISE SUMMARY:"""

CONCISE_SUMMARY_PROMPT = PromptTemplate(
    template=CONCISE_SUMMARY_PROMPT_TEMPLATE, 
    input_variables=["text"]
)  

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

QUESTION_VARIANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """Generate five variants of the following text that can be used as prompts for vectorstore lookup. 
Maintain the theme of the original. Do not number variants in your output. Output must be separated by newlines."""
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)