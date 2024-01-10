import os
from unittest.mock import MagicMock

import pytest
from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseMemory, LLMResult, Generation
from langchain.tools import BaseTool

from semterm.agent.TerminalAgent import TerminalAgent
from semterm.agent.TerminalAgentPrompt import PREFIX
from semterm.terminal.TerminalOutputParser import TerminalOutputParser


@pytest.fixture
def mock_tools():
    tools = [MagicMock(spec=BaseTool) for _ in range(3)]

    # Set custom name and description for each tool
    for idx, tool in enumerate(tools):
        tool.name = f"Tool{idx + 1}"
        tool.description = f"Tool{idx + 1} description"

    return tools


@pytest.fixture
def terminal_agent(mock_tools, monkeypatch):
    memory_mock = MagicMock(spec=BaseMemory)
    system_message = PREFIX.format(current_directory=os.getcwd())
    output_parser_mock = MagicMock(spec=TerminalOutputParser)
    verbose_mock = False

    def mock_generate_prompt(*args, **kwargs):
        return LLMResult(generations=[[Generation(text="Hello")]])

    llm_mock = MagicMock(spec=BaseLanguageModel)
    llm_mock.generate_prompt = mock_generate_prompt

    # Instantiate the TerminalAgent using the from_llm_and_tools method
    terminal_agent_instance = TerminalAgent.from_llm_and_tools(
        llm=llm_mock,
        tools=mock_tools,
        memory=memory_mock,
        system_message=system_message,
        output_parser=output_parser_mock,
        verbose=verbose_mock,
    )

    return terminal_agent_instance
