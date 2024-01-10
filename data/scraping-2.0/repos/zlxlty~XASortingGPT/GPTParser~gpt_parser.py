import asyncio
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Dict, List

import dotenv
from langchain import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

from GPTParser.define import Dict

from .define import *
from .model import (AttributeModel, CleaningPrompt, Message, MessagePrompt,
                    SortingPrompt)


class Parser(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError("Parse method not implemented")


class InterestParser(Parser):
    """Messages[34] to Messages[49]"""

    async def __call__(self, kwargs: Dict[str, Any]) -> str:
        if STR_MESSAGES not in kwargs:
            return ""
        interests = []
        messages = kwargs[STR_MESSAGES]
        for message in messages:
            if message.role == "user":
                interests += message.content.split(",")

        await asyncio.sleep(0)
        return ",".join(interests)


class GPTParser(Parser):
    templates = [Message(role=ROLE_USR, content=f"Hello XA! {{{STR_MESSAGES}}}")]

    def __init__(self):
        self.chat = self.create_chat()
        self.prompt = self.create_prompt()
        self.chain = self.create_chain()

    def create_chat(self):
        return ChatOpenAI(
            model="gpt-3.5-turbo-16k-0613",
            openai_api_key=dotenv.dotenv_values()["OPENAI_API_KEY"],
        )

    def create_prompt(self):
        from_template = {
            ROLE_USR: HumanMessagePromptTemplate.from_template,
            ROLE_BOT: AIMessagePromptTemplate.from_template,
            ROLE_SYS: SystemMessagePromptTemplate.from_template,
        }

        return ChatPromptTemplate.from_messages(
            [
                from_template[template.role](template.content)
                for template in self.templates
            ]
        )

    def create_chain(self):
        return LLMChain(llm=self.chat, prompt=self.prompt)

    @abstractmethod
    def is_kwargs_valid(self, kwargs: Dict[str, Any]):
        raise NotImplementedError("Prompt argument check not implemented")

    async def __call__(self, kwargs: Dict[str, str]) -> str | Dict[str, Any]:
        self.is_kwargs_valid(kwargs)
        return await self.chain.arun(**kwargs)


class CharacterGPTParser(GPTParser):
    """Messages[6] to Messages[33]"""

    templates = PROMPT_TEMPLATE[SEG_CHARACTER]

    def is_kwargs_valid(self, kwargs: Dict[str, Any]) -> bool:
        MessagePrompt(**kwargs)


class StoryGPTParser(CharacterGPTParser):
    """Messages[50] to Messages[63]"""

    templates = PROMPT_TEMPLATE[SEG_STORY]


class SortingResultParser(GPTParser):
    templates = PROMPT_TEMPLATE[STR_QUAL_JUDGEMENT]

    def is_kwargs_valid(self, kwargs: Dict[str, Any]) -> bool:
        SortingPrompt(**kwargs)


class FormattedOutputParser(GPTParser):
    templates = PROMPT_TEMPLATE[STR_GPT_FUNC_OUTPUT]

    def create_chain(self):
        return create_structured_output_chain(AttributeModel, self.chat, self.prompt)

    def is_kwargs_valid(self, kwargs: Dict[str, Any]):
        CleaningPrompt(**kwargs)


ParserMap = {
    SEG_INTEREST: InterestParser(),
    SEG_CHARACTER: CharacterGPTParser(),
    SEG_STORY: StoryGPTParser(),
    STR_QUAL_JUDGEMENT: SortingResultParser(),
    STR_GPT_FUNC_OUTPUT: FormattedOutputParser(),
}


async def aparse(parser_type: str, kwargs: Dict[str, Any]) -> str | Dict[str, Any]:
    # print(parser_type)
    if parser_type not in ParserMap:
        raise ValueError(f"Parser type {parser_type} not found")

    parser = ParserMap[parser_type]
    return await parser(kwargs)
