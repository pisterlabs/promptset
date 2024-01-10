from abc import ABC, abstractmethod
from typing import Generator

from reactpy.core.component import (
    Component as reactpy_Component,
    component as reactpy_component,
)
from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import BaseTool


class RendersToText(ABC):
    @abstractmethod
    def render_to_text(self) -> str:
        try:
            messages = self.render_to_messages()
            return "\n\n".join(message.content for message in messages)
        except NotImplementedError:
            pass
        raise NotImplementedError("`render_to_text` not implemented")


class RendersToMessages(ABC):
    @abstractmethod
    def render_to_messages(self) -> Generator[BaseMessage, None, None]:
        try:
            yield HumanMessage(content=self.render_to_text())
        except NotImplementedError:
            pass
        raise NotImplementedError("`render_to_messages` not implemented")


class HasLangchainTools(ABC):
    @abstractmethod
    @property
    def lc_tools(self) -> list[BaseTool]:
        raise NotImplementedError("`llm_tools` not implemented")


class RendersToReactPy(ABC):
    @abstractmethod
    @reactpy_component
    def render_to_reactpy(self) -> reactpy_Component:
        raise NotImplementedError("`render_to_reactpy` not implemented")
