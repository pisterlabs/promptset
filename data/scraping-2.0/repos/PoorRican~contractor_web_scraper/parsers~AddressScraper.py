from typing import ClassVar

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable

from llm import LONG_MODEL_PARSER, LLM
from parsers.TextSnippetScraper import TextSnippetScraper
from typedefs.address import Address


_address_parser = PydanticOutputParser(pydantic_object=Address)


def _address_scraper_chain() -> Runnable:
    """ Create a chain of LLM models to extract addresses from text snippets """
    _address_scaper_prompt = PromptTemplate.from_template(
        """You will be given the HTML content from a company website.
    
        Here is the content: ```{content}```
    
        What is the mailing address of the company?
        Return the properly formatted address separated by commas and nothing else.
        If there is no mailing address within the content, return 'no address' and nothing else.
        """
    )

    _formatter_prompt = PromptTemplate.from_template(
        """You will be given text that should directly represent a mailing address.

        Here is the address: {address}

        Return the address formatted as follows:
        {format_instructions}
        """,
        partial_variables={'format_instructions': _address_parser.get_format_instructions()},
    )

    _address_extract_chain: Runnable = _address_scaper_prompt | LONG_MODEL_PARSER
    return {'address': _address_extract_chain} | _formatter_prompt | LLM | _address_parser


class AddressScraper(TextSnippetScraper[Address]):
    _chain: ClassVar[Runnable] = _address_scraper_chain()
    _failure_text: ClassVar[str] = 'no address'
    _search_type: ClassVar[str] = 'address'
