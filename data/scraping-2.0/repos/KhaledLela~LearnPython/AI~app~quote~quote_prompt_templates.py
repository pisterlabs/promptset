from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import SystemMessage


def language_structure():
    return "1,en-US,4,sv,5,es,6,ar,7,fr,8,yo,9,de,10,ot,11,ta,12,pt-BR"


def create_quote_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that provide a relevant quotes."),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Provide a quote from the following provided content: 

        {source}

        Please provide a response quote that follows the output schema.""")
    ])
