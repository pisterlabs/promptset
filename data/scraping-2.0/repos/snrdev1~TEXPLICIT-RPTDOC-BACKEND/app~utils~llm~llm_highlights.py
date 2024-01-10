from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from app.utils.common import Common
from app.utils.llm.base import break_up_text


def generate_highlights(text):
    """
    The function `generate_highlights` takes in a text as input, breaks it up into chunks, and returns a
    list of highlights for each chunk.

    Args:
      text: The `text` parameter is the input text that you want to generate highlights from.

    Returns:
      The function `generate_highlights` returns a dictionary with the following keys:
    - "suc": A boolean value indicating whether the generation of highlights was successful or not.
    - "out": A list of highlights generated from the input text.
    - "st_time": The start time of the generation process.
    - "ed_time": The end time of the generation process.
    """
    try:
        print("Using LLM for highlight generation!")
        st_time = datetime.utcnow()
        highlights = []

        for chunk in break_up_text(text):
            highlights.extend(_get_highlight(chunk))

        ed_time = datetime.utcnow()

        if len(highlights) > 0:
            return {
                "suc": True,
                "out": highlights,
                "st_time": st_time,
                "ed_time": ed_time,
            }

        else:
            return {
                "suc": False,
                "out": highlights,
                "st_time": st_time,
                "ed_time": ed_time,
            }

    except Exception as e:
        Common.exception_details("generate_highlights", e)
        return []


def _get_highlight(text):
    """
    The function `_get_highlight` generates concise highlight sentences that capture key points and
    important keyphrases from the given text.
    
    Args:
      text: The `text` parameter is the input text that you want to generate highlights from. It should
    be a string containing the content you want to summarize.
    
    Returns:
      The function `_get_highlight` returns the highlights generated from the given text.
    """
    try:
        output_parser = _get_output_parser()
        format_instructions = output_parser.get_format_instructions()

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    """"
                    Summarize the text by generating concise highlight sentences (multiple) that capture key points and important keyphrases from the content. 
                    \nText : {question}\
                    \n{format_instructions}
                    """,
                )
            ],
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )

        model = ChatOpenAI(temperature=0, model=Config.FAST_LLM_MODEL)
        _input = prompt.format_prompt(question=text)
        output = model(_input.to_messages())
        highlights = output_parser.parse(output.content)

        return highlights["highlights"]

    except Exception as e:
        Common.exception_details("_get_highlight", e)
        return ""


def _get_output_parser():
    """
    The function `_get_output_parser` returns a structured output parser object that is created from a
    list of response schemas.

    Returns:
      The function `_get_output_parser` returns an instance of `StructuredOutputParser` if there are no
    exceptions. If an exception occurs, it returns an empty string.
    """
    try:
        response_schemas = [
            ResponseSchema(
                name="highlights",
                description="list containing multiple dictionaries of the form {'KeyPhrases': 'keyphrase in the sentence' and 'Sentence': 'highlight sentence'}",
            )
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        return output_parser

    except Exception as e:
        Common.exception_details("_get_output_parser", e)
        return ""
