from typing import *
from openai_function_call import openai_function, openai_schema
from boto3 import Session
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module
from ..config import credentials

session = Session(**credentials)


class SentimentResponse(BaseModel):
    Sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"] = Field(
        ...,
        description="The inferred sentiment that Amazon Comprehend has the highest level of confidence in.",
        alias="Sentiment",
    )
    SentimentScore: Dict[str, float] = Field(
        ...,
        description="The level of confidence that Amazon Comprehend has in the accuracy of its detection of the POSITIVE sentiment.",
        alias="SentimentScore",
    )


class Comprehend(BaseModel):
    Text: str = Field(..., description="Input text to analyze.", alias="Text")
    LanguageCode: str = Field(
        default="en",
        description="Optional language code for the Sentiment Analysis request.",
        alias="LanguageCode",
    )

    @classmethod
    def from_text(cls, text: str):
        return cls(Text=text)

    @property
    def client(self):
        return Session().client("comprehend", region_name="us-east-1")

    def detect_sentiment(self):
        response = self.client.detect_sentiment(**self.dict(exclude_none=True))
        return SentimentResponse(**response)
