"""Model utilities."""

from typing import Sequence
from colorama import Fore

from langchain.schema import BaseMessage
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI

precise_model = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview", verbose=False)  # type: ignore
creative_model = ChatOpenAI(temperature=0.7, model_name="gpt-4", verbose=False)  # type: ignore
# super_creative_model = ChatOpenAI(temperature=1.0, model_name="gpt-4", verbose=False)  # type: ignore
super_creative_model = ChatOpenAI(temperature=1.0, model_name="gpt-4-1106-preview", verbose=False)  # type: ignore
fast_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=False)  # type: ignore
broad_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", verbose=False)  # type: ignore
# anthropic models: https://docs.anthropic.com/claude/reference/selecting-a-model

def query_model(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    color: str = Fore.RESET,
    preamble: str | None = None,
    printout: bool = True,
) -> str:
    """Query an LLM chat model. `preamble` is printed before the result."""
    if preamble is not None and printout:
        print(f"\033[1;34m{preamble}\033[0m")
    result = model(list(messages)).content
    if printout:
        print(f"{color}{result}{Fore.RESET}")
    return result
