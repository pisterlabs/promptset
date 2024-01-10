import abc
import re
from enum import Enum

import openai
from loguru import logger
from pydantic import BaseModel

from springtime.models.open_ai import OpenAIModel


class TrafficlightAnswer(str, Enum):
    red = "red"
    yellow = "yellow"
    green = "green"


class ScanResult(BaseModel):
    description: str
    tags: list[str]
    is_financial_document: TrafficlightAnswer
    is_cim: TrafficlightAnswer


class ScanService(abc.ABC):
    @abc.abstractmethod
    def scan(
        self,
        *,
        file_name: str,
        text: str,
    ) -> ScanResult:
        pass


LIMIT = 7000


PROMPT = """

You are an expert financial analyst.

You will be given an excerpt from a document.

* Provide a 1 line description of the entire document. Do not start with "This document is about..." or "This document describes..." or "Excerpt from ..." just describe the document in 1 line.
* Provide tags describing the category of the document
* Is this is a financial document? Reply with green if you are very sure it is financial document. Reply with yellow if you are not sure. Reply with red if you are very sure it is not a financial document.
* Is this is document a  Confidential Information Memorandum or Investor Prospectus or Management Presentation? Reply with green if you are very sure it is financial document. Reply with yellow if you are not sure. Reply with red if you are very sure it is not a financial document.

Output your reponse in the following format:

Description: <description>
Tags: <tag1>, <tag2>, <tag3>
Is financial document: <green/yellow/red>
Is confidential information memorandum: <green/yellow/red>
"""


class OpenAIScanService(ScanService):
    def __init__(self, model: OpenAIModel) -> None:
        self.model = model

    def scan(
        self,
        *,
        file_name: str,
        text: str,
    ) -> ScanResult:
        processed_text = first_chunk(text, LIMIT)
        with_out_white_space = remove_extra_whitespace(processed_text)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""
file name: {file_name}
file excerpt: {with_out_white_space}
                 """,
                },
            ],
            temperature=0,
        )
        choices = response["choices"]
        if len(choices) == 0:
            logger.warning("No choices returned from OpenAI")
        first_choice = choices[0]
        description = first_choice["message"]["content"]
        return parse_response(description)


def parse_response(response: str) -> ScanResult:
    description = response.find("Description:")
    tags = response.find("Tags:")
    fin_document = response.find("Is financial document:")
    cim = response.find("Is confidential information memorandum:")

    description = response[description:tags].split("Description:")[1].strip()
    tag_string = response[tags:fin_document].split("Tags:")[1].strip()
    fin_document_string = (
        response[fin_document:cim].split("Is financial document:")[1].strip().lower()
    )
    cim_string = (
        response[cim:]
        .split("Is confidential information memorandum:")[1]
        .strip()
        .lower()
    )

    final_tag_string = tag_string.split("Is financial document:")[0].strip()
    tags = sorted({tag.strip() for tag in tag_string.split(",")})
    return ScanResult(
        description=description,
        tags=tags,
        is_financial_document=TrafficlightAnswer(fin_document_string),
        is_cim=TrafficlightAnswer(cim_string),
    )


def remove_extra_whitespace(s: str) -> str:
    return re.sub(r"\n+", "\n", s.replace("\n ", "\n")).strip()


def first_chunk(s: str, maxlength: int):
    gen = get_chunks(s, maxlength)
    return next(gen)


def get_chunks(s: str, maxlength: int):
    start = 0
    end = 0
    while start + maxlength < len(s) and end != -1:
        end = s.rfind(" ", start, start + maxlength + 1)
        yield s[start:end]
        start = end + 1
    yield s[start:]
