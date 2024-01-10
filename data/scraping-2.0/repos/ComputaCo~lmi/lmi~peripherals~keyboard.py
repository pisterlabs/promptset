from langchain.tools import BaseTool, tool
from lmi.components.abstract.component import Component
from lmi.handlers import KeyboardInputEventHandler, KeypressEventHandler
from lmi.peripherals.base import Peripheral


class Keyboard(Peripheral):
    def lc_tools(self, component: Component) -> list[BaseTool]:
        return [
            KeyboardInputEventHandler.lc_tool(component),
            KeypressEventHandler.lc_tool(component),
        ]
