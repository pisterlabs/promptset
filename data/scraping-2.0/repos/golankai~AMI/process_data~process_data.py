from dataclasses import dataclass
from typing import Optional
from enum import Enum

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class MessageType(Enum):
    HUMAN = 0
    SYSTEM = 1


@dataclass
class QueryData:
    template: str
    parser: PydanticOutputParser
    type: Optional[MessageType] = MessageType.HUMAN

    def get_prompt(self) -> HumanMessagePromptTemplate | SystemMessagePromptTemplate:
        template, parser = self.template, self.parser
        prompt = None
        if self.type == MessageType.HUMAN:
            prompt = HumanMessagePromptTemplate.from_template(
                template=template, output_parser=parser
            )
        if self.type == MessageType.SYSTEM:
            prompt = SystemMessagePromptTemplate.from_template(
                template=template, output_parser=parser
            )

        return prompt


@dataclass
class ProcessData:
    base: PromptTemplate
    queries: list[QueryData]

    def get_base_template(self) -> PromptTemplate:
        return self.base
