"""Tests for ZBaseTool."""

from langchain_contrib.tools import TerminalTool, ZBaseTool


def test_wrapper() -> None:
    """Test that a ZBaseTool can be wrapped around an existing tool."""
    tool = ZBaseTool.from_tool(base_tool=TerminalTool(), color="red")
    assert tool.color == "red"


class DemoTool(ZBaseTool):
    """A demonstration of subclassing ZBaseTool."""

    name: str = "Demo"
    description: str = "Demo tool"

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return f"Demonstration: {tool_input}"


def test_subclass() -> None:
    """Test that a ZBaseTool can be subclassed."""
    tool = DemoTool(color="pink")
    assert tool.color == "pink"
    assert tool.run("This is pink") == "Demonstration: This is pink"
