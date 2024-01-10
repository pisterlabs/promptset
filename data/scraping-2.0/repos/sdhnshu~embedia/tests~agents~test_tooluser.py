from copy import deepcopy

import pytest
from embedia import Persona
from embedia.agents import ToolUserAgent
from embedia.tools import PythonInterpreterTool, TerminalTool

from tests.core.definitions import OpenAIChatLLM


@pytest.mark.asyncio
async def test_tool_user(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    tool_user = ToolUserAgent(
        chatllm=OpenAIChatLLM(), tools=[PythonInterpreterTool(), TerminalTool()]
    )
    await tool_user("List all files in this directory")


@pytest.mark.asyncio
async def test_tool_user_timeout(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    tool_user = ToolUserAgent(
        chatllm=OpenAIChatLLM(),
        tools=[PythonInterpreterTool(), TerminalTool()],
        max_duration=1,
    )
    await tool_user("List all files in this directory")


@pytest.mark.asyncio
async def test_tool_user_with_llm(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")

    python_coder = deepcopy(OpenAIChatLLM())
    await python_coder.set_system_prompt(
        Persona.CodingLanguageExpert.format(language="Python")
    )

    code = await python_coder(
        "Count number of lines of python code in the current directory"
    )

    tool_user = ToolUserAgent(chatllm=OpenAIChatLLM(), tools=[PythonInterpreterTool()])
    await tool_user(code)
