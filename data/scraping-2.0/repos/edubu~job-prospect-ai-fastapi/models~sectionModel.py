from typing import List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field

from config import OPENAI_API_KEY


# Output data structure
class PageSection(BaseModel):
    section: str = Field(description="The section details in markdown format")


# output parser
parser = PydanticOutputParser(pydantic_object=PageSection)
# retry_parser = RetryWithErrorOutputParser.from_llm(
#     parser=parser, llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
# )

# Prompt Templates
SECTION_SYSTEM_TEMPLATE = "You are a analyst assistant. You take summaries from web pages and create a detailed section for the company summary."
SECTION_HUMAN_TEMPLATE = """\
You are a analyst that writes reports on companies
You are creating a concise and detailed section for the {section_name} section of the company summary.

For example, if you are creating a section for the Company History section of the company summary, you will write a detailed section about the company history.
This section will include details such as when the company was founded, who founded the company, and any other details that are important to the company history.

Use the following rules when constructing the section:
- Write everything in markdown format(e.g use <br> instead of \\n)
- Keep the length of the section to no more than 500 words
- Any information that can be conveyed in bullet points should be conveyed in bullet points
- Include "# {section_name}" at the beggining of your answer
- Use any knowledge you currently have about the company, industry, and other relevant information

I will provide the page summaries that you will use as external sources to write this section.

Page Summaries:\n 
{page_summaries}

{format_instructions}

Answer: 
"""

# Prompts
system_prompt_template = SystemMessagePromptTemplate.from_template(
    SECTION_SYSTEM_TEMPLATE
)
human_prompt_template_original = PromptTemplate(
    template=SECTION_HUMAN_TEMPLATE,
    input_variables=["section_name", "page_summaries"],
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
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k", temperature=0
)

# Chains
section_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=parser)
