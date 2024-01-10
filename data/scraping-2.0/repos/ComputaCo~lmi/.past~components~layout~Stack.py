from typing import Generator, List
from langchain.schema import BaseMessage
from pydantic import BaseModel
from lmi.components.Component import Component
from lmi.utils.lmi_message import LMIMessage


class Stack(Component):
    separator: str or None = "\n"

    _components: List[Component] = []

    def __init__(
        self,
        size: int = None,
        name: str = None,
        separator: str or None = separator,
        components: List[Component] = _components,
    ):
        super().__init__(
            size=size, name=name, separator=separator, _components=components
        )

    @property
    def children(self) -> list[Component]:
        return self._components or []

    def llm_render(self) -> str:
        return (self.separator or "").join(
            [component.render_llm() for component in self.children]
        )

    def render_messages_llm(self) -> Generator[BaseMessage, None, None]:
        for component in self.children:
            yield from component.render_messages_llm()
            if self.separator:
                yield LMIMessage(content=self.separator)
