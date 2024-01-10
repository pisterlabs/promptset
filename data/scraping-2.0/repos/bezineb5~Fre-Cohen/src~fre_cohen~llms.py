import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Type

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseOutputParser

from fre_cohen.configuration import Config

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 5


# Enumeration to choose speed or accuracy
class LLMQualityEnum(str, Enum):
    """Enumeration of the LLM quality"""

    SPEED = "speed"
    ACCURACY = "accuracy"
    VISION = "vision"


def get_openai_llm(config: Config, quality: LLMQualityEnum) -> ChatOpenAI:
    """Returns the OpenAI LLM"""

    max_tokens = None
    response_format = None

    # Choose the model
    if quality == LLMQualityEnum.SPEED:
        model = "gpt-3.5-turbo-1106"
        response_format = {"type": "json_object"}
    elif quality == LLMQualityEnum.ACCURACY:
        model = "gpt-4-1106-preview"
        response_format = {"type": "json_object"}
    elif quality == LLMQualityEnum.VISION:
        model = "gpt-4-vision-preview"
        max_tokens = 256
        # response_format = {"type": "json_object"}
    else:
        raise ValueError(f"Unknown quality: {quality}")

    # Timeout is in seconds
    llm_chain = ChatOpenAI(
        api_key=config.openai_api_key,
        model=model,
        timeout=config.request_timeout_seconds,
        max_tokens=max_tokens,
    )
    if response_format:
        llm_chain = llm_chain.bind(response_format=response_format)

    return llm_chain


def get_llm(config: Config, quality: LLMQualityEnum) -> BaseChatModel:
    """Returns the LLM"""
    return get_openai_llm(config, quality=quality)


def build_llm_chain(
    config: Config,
    pydantic_message: Type[BaseModel],
    prompts: list[BaseMessagePromptTemplate],
    quality: LLMQualityEnum = LLMQualityEnum.SPEED,
) -> LLMChain:
    """Builds the LLM chain"""
    return _JsonLLMChain(
        config=config, pydantic_message=pydantic_message, prompts=prompts
    ).llm_chain(quality=quality)


class _JsonLLMChain:
    def __init__(
        self,
        config: Config,
        pydantic_message: Type[BaseModel],
        prompts: list[BaseMessagePromptTemplate],
    ) -> None:
        super().__init__()
        self._config = config
        self._pydantic_message = pydantic_message
        self._prompts = prompts

    def _prompt_template(
        self, parser: BaseOutputParser, prompts: list[BaseMessagePromptTemplate]
    ) -> ChatPromptTemplate:
        """Returns the prompt template"""
        all_prompts = [] if not prompts else list(prompts)

        all_prompts.append(
            SystemMessagePromptTemplate.from_template("{format_instructions}")
        )

        return ChatPromptTemplate.from_messages(all_prompts).partial(
            format_instructions=parser.get_format_instructions()
        )

    def _output_parser(self) -> BaseOutputParser:
        """Returns the pydantic output parser"""
        return PydanticOutputParser(
            pydantic_object=self._pydantic_message,
        )

    def llm_chain(self, quality: LLMQualityEnum) -> LLMChain:
        """Returns the LLM chain"""
        parser = self._output_parser()
        return LLMChain(
            llm=get_llm(self._config, quality=quality),
            prompt=self._prompt_template(parser, self._prompts),
            output_parser=parser,
        )


def retry_on_error(
    func: Callable, max_retries: int = DEFAULT_RETRIES, sleep_time: float = 1.0
) -> Callable:
    """Decorator to retry a function on error"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function"""
        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning("Error in function %s: %s", func, e)
                time.sleep(sleep_time)
        raise RuntimeError(f"Function {func} failed after {max_retries} retries")

    return wrapper
