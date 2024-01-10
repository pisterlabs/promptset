from typing import List

from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class TopicOutput(BaseModel):
    title: str = Field(description="title of the new topic")
    description: str = Field(description="Description of what about the questions are")
    questions: List[str] = Field(description="list of questions of this topic")


class TopicParserSingleton:
    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = PydanticOutputParser(pydantic_object=TopicOutput)
        return cls.__instance