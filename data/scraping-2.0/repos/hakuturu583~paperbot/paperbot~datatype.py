from langchain.pydantic_v1 import BaseModel, Field, validator

from typing import List

from enum import Enum


class MetaData(BaseModel):
    title: str = Field(description="Title of the paper.")
    authors: List[str] = Field(description="List of the author's names")


class UserAction(Enum):
    SUMMARY_PAPER = "Summary paper"
    ANSWER_QUESTION_FROM_PAPER = "Answer question from paper"


class Language(Enum):
    ENGLISH = "english"
    JAPANESE = "japanese"
