from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from lmi.components.abstract.component import Component
from lmi.handlers import (
    AdvancedClickEventHandler,
    ClickEventHandler,
    HoverEventHandler,
    ScrollEventHandler,
)
from lmi.peripherals.base import Peripheral


class Mouse(Peripheral):
    scroll_wheel = True
    advanced = False
    hover = True

    def lc_tools(self, component: Component) -> list[BaseTool]:
        tools = []
        if not self.advanced:
            tools.append(ClickEventHandler.lc_tool(component))
        else:
            tools.append(AdvancedClickEventHandler.lc_tool(component))

        if self.scroll_wheel:
            tools.append(ScrollEventHandler.lc_tool(component))
        if self.hover:
            tools.append(HoverEventHandler.lc_tool(component))

        return tools
