from typing import List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field, validator

from config import OPENAI_API_KEY


# Output data structure
class PageCleanup(BaseModel):
    cleaned_content: str = Field(
        description="The cleaned up markdown of the company summary"
    )


# output parser
parser = PydanticOutputParser(pydantic_object=PageCleanup)

# Prompt Templates
CLEANUP_SYSTEM_TEMPLATE = "You are a writers assistant. You take company summaries and fix the grammatical errors, remove information that is often repeated, and format it in markdown for better visual appeal."
CLEANUP_HUMAN_TEMPLATE = """\
You are a writing assistant bot.
You take in a company summary in markdown format and fix mistakes such as grammar, spelling, or improve the markdown format.
For example, if there is a header that contains "#" and is not at the start of the sentence, you know that it won't render as a header correctly in markdown, so you will remove anything before the "#".
Keep all of the content you can, but make sure that the markdown is correct.
If some phrases are repeated too often, you will remove or reword them in a different way.

{format_instructions}\n

Company Summary: 
{company_summary}

Answer:
"""

# Prompts
system_prompt_template = SystemMessagePromptTemplate.from_template(
    CLEANUP_SYSTEM_TEMPLATE
)
human_prompt_template_original = PromptTemplate(
    template=CLEANUP_HUMAN_TEMPLATE,
    input_variables=["company_summary"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
human_prompt_template = HumanMessagePromptTemplate(
    prompt=human_prompt_template_original
)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_prompt_template, human_prompt_template]
)


# LlMs
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k", temperature=0.3
)

# Chains
cleanup_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=parser)
