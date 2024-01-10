import importlib
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from typing import cast

from langchain.chains import LLMChain


class RegisteredLLMMeta(ABCMeta):
    registry: dict[str, "RegisteredLLM"] = {}

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not dct.get("_base_class", False):
            cls = cast("RegisteredLLM", cls)
            if cls.get_cli_name() in cls.registry:
                raise ValueError(f"Duplicate LLM name: {cls.get_cli_name()}")
            cls.registry[cls.get_cli_name()] = cls


class RegisteredLLM(metaclass=RegisteredLLMMeta):
    _base_class = True
    supports_parallel = True

    @classmethod
    @abstractmethod
    def get_llm(cls, system_prompt: str) -> LLMChain:
        """Create an instance of an LLM with the given system prompt."""
        ...

    @classmethod
    @abstractmethod
    def get_cli_name(cls) -> str:
        """The name for this LLM in the CLI."""
        ...


def get_llm(llm_name: str, system_prompt: str) -> LLMChain:
    """Fetch the LLM model based on its plaintext name."""
    llm = RegisteredLLMMeta.registry.get(llm_name)
    if llm is None:
        raise ValueError(f"Unknown LLM name: {llm_name}")
    return llm.get_llm(system_prompt)


def register_llms() -> None:
    """Imports every file in this directory to register the LLMs."""
    for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
        importlib.import_module(f".{name}", __package__)
