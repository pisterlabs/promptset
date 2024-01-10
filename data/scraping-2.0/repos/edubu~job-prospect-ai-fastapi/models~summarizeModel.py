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
class PageSummary(BaseModel):
    summary: str = Field(description="The summary of the page in 2-3 paragraphs")
    sections: List[str] = Field(description="The sections that this page is useful for")


# output parser
parser = PydanticOutputParser(pydantic_object=PageSummary)

# Prompt Templates
SUMMARIZE_SYSTEM_TEMPLATE = "You are a page summarizer assitant. You take text extracted from a web page and summarize the information on the web page. You also decide which sections of company summary this page will be useful for."
SUMMARIZE_HUMAN_TEMPLATE = """\
You are a text summarization bot.
Below is the text scraped from ${page_url}.
{page_text}

Summarize this text to fit within 500 words. Summarize the text while keeping specific details and facts about the company.
As Examples:
- if you are summarizing a url of https://epic.com/solutions/eletronic-health-records, your summary will be explaining the solution of electronic health records that the company epic provides.
- if you are summarizing a url of https://www.athenahealth.com/careers/locations, your summary will include a bulleted list of the locations that the company athenahealth has offices in.
- if you are summarizing a url of https://www.athenahealth.com/about/who-we-are, your summary will include facts and details about the company athenhealth. Such as when it was founded, their culture, etc.
- if you are summarizing a url of https://www.airtable.com/about, your summary will include specific facts about where the offices are located, when it was founded, contact emails, and other details about the company.
- if you are summarizing a url of https://www.airtable.com/pricing, your summary will include specific details about the pricing of all of their products.

In addition you should decide which sections of company summary this page will be useful.
Take into account what existing knowledge you have about the website/company.
For Example:
When deciding if the page summary should be in the "Key Competitors" section, if you don't know anything about the website, then knowldge about the industry will be useful.
Therefore, you may add the /products page to the "Key Competitors" section.

Choose from the following sections:
- Company Summary
- Products and Services
- Business Model
- Target Audience
- Key Competitors
- Contact Information and Company Details

{format_instructions}\n
"""

# Prompts
system_prompt_template = SystemMessagePromptTemplate.from_template(
    SUMMARIZE_SYSTEM_TEMPLATE
)
human_prompt_template_original = PromptTemplate(
    template=SUMMARIZE_HUMAN_TEMPLATE,
    input_variables=["page_url", "page_text"],
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
summarize_chain = LLMChain(llm=llm, prompt=chat_prompt, output_parser=parser)
