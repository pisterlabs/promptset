import os

import pytest
from unittest.mock import MagicMock

from langchain.base_language import BaseLanguageModel
from langchain.prompts import SystemMessagePromptTemplate
from langchain.tools import BaseTool
from langchain.schema import (
    AgentAction,
    BaseMessage,
    AIMessage,
    SystemMessage,
    BaseMemory,
)
from semterm.agent.TerminalAgent import TerminalAgent
from semterm.agent.TerminalAgentPrompt import PREFIX, SUFFIX
from semterm.terminal.TerminalOutputParser import TerminalOutputParser


class TestTerminalAgent:
    def test_create_prompt(self, terminal_agent, mock_tools):
        system_message = PREFIX.format(current_directory=os.getcwd())
        human_message = SUFFIX
        input_variables = ["input", "chat_history", "agent_scratchpad"]

        prompt = TerminalAgent.create_prompt(
            tools=mock_tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
        )
        # Extract properties from the returned ChatPromptTemplate
        system_message_from_prompt = prompt.messages[0].format_messages()[0].content
        human_message_from_prompt = (
            prompt.messages[2].format_messages(input="test input")[0].content
        )

        # Assert that the properties have the expected values
        assert system_message_from_prompt == system_message
        assert all(tool.name in human_message_from_prompt for tool in mock_tools)
        assert all(tool.description in human_message_from_prompt for tool in mock_tools)
        assert prompt.input_variables == input_variables

    def test_construct_scratchpad(self, terminal_agent):
        intermediate_steps = [
            (AgentAction(tool="Human", tool_input="cd ..", log="cd .."), ""),
            (
                AgentAction(
                    tool="TerminalTool", tool_input="ls", log="ls command executed"
                ),
                "file1 file2",
            ),
            (
                AgentAction(
                    tool="TerminalTool",
                    tool_input=["cd ..", "ls"],
                    log="['cd ..', 'ls']",
                ),
                "file1 file2",
            ),
        ]

        scratchpad = terminal_agent._construct_scratchpad(intermediate_steps)

        assert isinstance(scratchpad, list)
        assert all(isinstance(msg, BaseMessage) for msg in scratchpad)
        assert len(scratchpad) == 5
        assert isinstance(scratchpad[0], AIMessage)
        assert scratchpad[0].content == "cd .."
        assert isinstance(scratchpad[1], AIMessage)
        assert scratchpad[1].content == "ls command executed"
        assert isinstance(scratchpad[2], SystemMessage)
        assert scratchpad[2].content.startswith("Observation:")
        assert isinstance(scratchpad[3], AIMessage)
        assert scratchpad[3].content == "['cd ..', 'ls']"
