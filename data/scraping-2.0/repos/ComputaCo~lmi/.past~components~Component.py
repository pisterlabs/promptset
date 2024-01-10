from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, Generator
from langchain.schema import BaseMessage
from pydantic import BaseModel
from langchain.tools import BaseTool
from lmi.abstract.human_interface import HumanCanInteractWithMixin, HumanCanViewMixin

from lmi.abstract.llm_interface import LLMCanInteractWithMixin, LLMCanViewMixin
from lmi.handlers.advanced_keyboard_event_handler import AdvancedKeyboardEventHandler
from lmi.handlers.click_event_handler import ClickEventHandler

from lmi.handlers.display_event_handler import DisplayEventHandler
from lmi.handlers.drag_event_handler import DragEventHandler
from lmi.handlers.drop_event_handler import DropEventHandler
from lmi.handlers.event_handler import Event, EventHandler
from lmi.handlers.focus_event_handler import FocusEventHandler
from lmi.handlers.hover_event_handler import HoverEventHandler
from lmi.handlers.keyboard_event_handler import KeyboardEventHandler
from lmi.handlers.mouse_event_handler import BaseMouseEventHandler
from lmi.handlers.scroll_event_handler import ScrollEventHandler
from lmi.utils.json_serializable import JSONSerializable
from lmi.utils.name_generator import HasUniqueNameMixin


class Component(
    DisplayEventHandler,
    LLMCanInteractWithMixin,
    LLMCanViewMixin,
    HumanCanViewMixin,
    HumanCanInteractWithMixin,
    HasUniqueNameMixin,
    JSONSerializable,
    BaseModel,
):
    size: int or None = None
    parent: Component or None = None
    sep: ClassVar[str] = "\n\n"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.post_init()

    def post_init(self):
        for child in self.children:
            child.parent = self

    @property
    def children(self) -> list[Component]:
        return []

    @property
    def visible_children(self) -> list[Component]:
        return self.children

    def render_llm(self) -> str:
        return self.sep.join(
            [component.render_llm() for component in self.visible_children]
        )

    def render_messages_llm(self) -> Generator[BaseMessage, None, None]:
        for child in self.visible_children:
            yield from child.render_messages_llm()

    @property
    def llm_tools(self) -> list[BaseTool]:
        return [tool for component in self.children for tool in component.llm_tools]

    def add_event_handler(self, capture=True, bubble=True):
        def wrapper(fn):
            ...

        return wrapper
