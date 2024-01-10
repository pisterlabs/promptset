from __future__ import annotations

from abc import ABC, abstractmethod
from langchain.tools import BaseTool

from pydantic import BaseModel, Field, validator
from abc import abstractmethod
from typing import Generator

from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import BaseTool
from reactpy.core.component import (
    Component as reactpy_Component,
    component as reactpy_component,
)
from reactpy import html

# from reactpy.core.component import Component as reactpy_Component, component as reactpy_component

from lmi.utils.interfaces import (
    HasLangchainTools,
    RendersToMessages,
    RendersToText,
    RendersToReactPy,
)
from lmi.handlers import DisplayEventHandler, EventHandler
from lmi.utils.misc import PASS, Handler, gen_unique_name
from lmi.utils.python import HasPostInit


class Component(
    RendersToReactPy,
    HasLangchainTools,
    RendersToMessages,
    RendersToText,
    HasPostInit,
    BaseModel,
    ABC,
):
    name: str | None = Field(None, description="The name of the component.")
    description: str | None = None
    parent: Component | None = Field(None, init=False, repr=False)
    # FIXME: implement the following fields into render methods
    # chars_priority: str | None = Field(None, description="The spatial allocation priority given to this component.")
    # min_chars: int | None = Field(None, description="The minimum number of characters this component should take up.")
    # max_chars: int | None = Field(None, description="The maximum number of characters this component should take up.")

    @validator("name")
    def check_name(cls, v):
        return (v or gen_unique_name(cls.__name__)).replace(" ", "_")

    @property
    def qualified_name(self) -> str:
        return self.parent.qualified_name + "." + self.name

    _children: list[Component] = Field([], init=False, repr=False)

    @property
    @abstractmethod
    def children(self) -> list[Component]:
        # declared as property rather than attr because not all components have children
        return normalize(self._children)

    @children.setter
    def children(self, value: list[Component]):
        self._children = value

    @property
    def children_recursive(self) -> list[Component]:
        children = []
        for child in self.children:
            children.append(child)
            children.extend(child.children_recursive)
        return children

    def get_child_by_name(self, name: str) -> Component:
        for child in self.children:
            if child.name == name:
                return child
        raise ValueError(f"no child with name {name}")

    def get_child_by_name_recursive(self, name: str) -> Component:
        for child in self.children_recursive:
            if child.name == name:
                return child
        raise ValueError(f"no child with name {name}")

    _enabled: bool = Field(
        True, description="Whether this component is enabled.", init=False, repr=False
    )
    # FIXME: how can i get `_enabled` to show up as `enabled` in the init signature and obj repr?

    @property
    def enabled(self) -> bool:
        return self.parent.enabled and self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        old_value, new_value = self.enabled, value
        self._enabled = value
        match old_value, new_value:
            case True, False:
                if self.focused:
                    self.lose_focus()
                self.on_disable()
            case False, True:
                self.on_enable()

    @property
    def disabled(self) -> bool:
        return not self.enabled

    @disabled.setter
    def disabled(self, value: bool):
        self.enabled = not value

    def toggle_enabled(self):
        self.enabled = not self.enabled

    on_enable: Handler = PASS
    on_disable: Handler = PASS

    @property
    def enabled_children(self) -> list[Component]:
        return [child for child in self.children if child.enabled]

    _visible: bool = Field(
        True, description="Whether this component is visible.", init=False, repr=False
    )
    # FIXME: how can i get `_visible` to show up as `visible` in the init signature and obj repr?

    @property
    def visible(self) -> bool:
        return self.parent.visible and self._visible

    @visible.setter
    def visible(self, value: bool):
        old_value, new_value = self.visible, value
        self._visible = value
        match old_value, new_value:
            case True, False:
                self.on_hide()
            case False, True:
                self.on_show()

    @property
    def hidden(self) -> bool:
        return not self.visible

    @hidden.setter
    def hidden(self, value: bool):
        self.visible = not value

    on_show: Handler = PASS
    on_hide: Handler = PASS

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def toggle_visibility(self):
        self.visible = not self.visible

    @property
    def visible_children(self) -> list[Component]:
        return [child for child in self.children if child.visible]

    _focusable: bool = Field(
        False, description="Whether this component can capture the focus."
    )

    @property
    def focusable(self) -> bool:
        return self.parent.enabled and self._focusable and self.enabled

    @focusable.setter
    def focusable(self, value: bool):
        old_val, new_val = self.focusable, value
        self._focusable = value

        match old_val, new_val:
            case True, False:
                self.lose_focus()
            case False, True:
                self.on_focusable()

    focused_child: Component | None = Field(None, init=False, repr=False)

    def get_focus(self):
        self.parent._focus_child(self)

    def lose_focus(self):
        self.parent._unfocus_child()

    def toggle_focus(self):
        self.set_focus(not self.focused)

    def set_focus(self, focused: bool):
        if focused:
            self.get_focus()
        else:
            self.lose_focus()

    @property
    def focused(self) -> bool:
        if not self.parent:
            return False
        if not self.focusable:
            return False
        return self.parent.focused_child is self

    @property
    def focusable_children(self) -> list[Component]:
        return [child for child in self.children if child.focusable]

    def _focus_child(self, child: Component):
        self.focused_child = child

    def _unfocus_child(self):
        self.focused_child = None

    def shift_focus(self, direction: int = 1):
        if not self.focused_child and not self.focusable_children:
            self.focused_child = None
            return
        if not self.focused_child and self.focusable_children:
            self.focused_child = self.focusable_children[0]
            return
        focused_index = self.focusable_children.index(self.focused_child)
        new_focused_index = (focused_index + direction) % len(self.focusable_children)
        self.focused_child = self.focusable_children[new_focused_index]

    def render_to_text(self):
        if not self.visible:
            return ""
        return self._render_to_text()

    def _render_to_text(self):
        return "\n\n".join(message.content for message in self.render_to_messages())

    def render_to_messages(self):
        if not self.visible:
            return
        yield from self._render_to_messages()
        if self.description:
            yield HumanMessage(f"({self.description})")

    def _render_to_messages(self):
        for child in self.children:
            yield from child.render_to_messages()

    def render_to_reactpy(self) -> reactpy_Component:
        if not self.visible:
            return
        return self._render_to_reactpy()

    @reactpy_component
    def _render_to_reactpy(self) -> reactpy_Component:
        return html.div([html.span(m.content) for m in self.render_to_messages()])

    @property
    def lc_tools(self) -> list[BaseTool]:
        return []
