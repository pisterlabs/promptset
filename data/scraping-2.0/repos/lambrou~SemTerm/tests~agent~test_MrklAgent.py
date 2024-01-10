from functools import partial

import pytest
from unittest.mock import MagicMock, patch

from langchain.agents import Agent, AgentExecutor
from langchain.base_language import BaseLanguageModel
from langchain.schema import BaseMemory
from langchain.tools import BaseTool

from semterm.agent.TerminalAgent import TerminalAgent
from semterm.agent.TerminalAgentExecutor import TerminalAgentExecutor
from semterm.config.Config import Config
from semterm.agent.MrklAgent import MrklAgent
from semterm.langchain_extensions.tools import MistakeTool
from semterm.terminal.TerminalOutputParser import TerminalOutputParser
from semterm.terminal.TerminalTool import TerminalTool


class TestMrklAgent:
    @pytest.fixture
    def mrkl_agent(self, monkeypatch):
        config_mock = MagicMock(spec=Config)
        config_parser_mock = MagicMock()
        config_parser_mock.getboolean.return_value = False
        config_parser_mock.getint.return_value = 10
        config_mock.get.return_value = config_parser_mock

        # Store the original methods as attributes of the instance
        original_load_tools = MrklAgent.load_tools
        original_initialize_memory = MrklAgent.initialize_memory
        original_initialize_agent = MrklAgent.initialize_agent
        original_initialize_executor = MrklAgent.initialize_executor

        monkeypatch.setattr(MrklAgent, "load_tools", MagicMock())
        monkeypatch.setattr(MrklAgent, "initialize_memory", MagicMock())
        monkeypatch.setattr(MrklAgent, "initialize_agent", MagicMock())
        monkeypatch.setattr(MrklAgent, "initialize_executor", MagicMock())

        chat_openai_mock = MagicMock()
        monkeypatch.setattr("semterm.agent.MrklAgent.ChatOpenAI", chat_openai_mock)

        agent = MrklAgent(config=config_mock)

        agent.original_load_tools = partial(original_load_tools, agent)
        agent.original_initialize_memory = partial(original_initialize_memory, agent)
        agent.original_initialize_agent = partial(original_initialize_agent, agent)
        agent.original_initialize_executor = partial(
            original_initialize_executor, agent
        )

        return agent

    def test_load_tools(self, mrkl_agent, monkeypatch):
        load_tools_mock = MagicMock(return_value=[MagicMock(spec=BaseTool)])
        terminal_tool_mock = MagicMock(spec=TerminalTool)
        mistake_tool_mock = MagicMock(spec=MistakeTool)
        monkeypatch.setattr("langchain.agents.load_tools", load_tools_mock)
        monkeypatch.setattr(
            "semterm.terminal.TerminalTool.TerminalTool", terminal_tool_mock
        )
        monkeypatch.setattr(
            "semterm.langchain_extensions.tools.MistakeTool", mistake_tool_mock
        )

        tools = mrkl_agent.original_load_tools()

        assert isinstance(tools, list)
        assert all(isinstance(tool, BaseTool) for tool in tools)
        assert any(isinstance(tool, TerminalTool) for tool in tools)

    def test_initialize_memory(self, mrkl_agent, monkeypatch):
        base_language_model_mock = MagicMock(spec=BaseLanguageModel)
        monkeypatch.setattr(
            "langchain.base_language.BaseLanguageModel",
            MagicMock(return_value=base_language_model_mock),
        )

        # Set the MagicMock instance as the 'llm' attribute of mrkl_agent
        mrkl_agent.llm = base_language_model_mock

        memory = mrkl_agent.original_initialize_memory()

        assert isinstance(memory, BaseMemory)

    def test_initialize_agent(self, mrkl_agent, monkeypatch):
        # Mock the objects used by the method
        base_language_model_mock = MagicMock(spec=BaseLanguageModel)
        terminal_agent_mock = MagicMock(spec=TerminalAgent)
        terminal_output_parser_mock = MagicMock(spec=TerminalOutputParser)

        # Set the MagicMock instances as the attributes of mrkl_agent
        mrkl_agent.llm = base_language_model_mock
        mrkl_agent.tools = [MagicMock(spec=BaseTool)]
        mrkl_agent.memory = MagicMock(spec=BaseMemory)
        mrkl_agent.verbose = False

        # Mock the constructors and methods
        monkeypatch.setattr(
            "langchain.base_language.BaseLanguageModel",
            MagicMock(return_value=base_language_model_mock),
        )
        monkeypatch.setattr(
            "semterm.agent.TerminalAgent.TerminalAgent.from_llm_and_tools",
            MagicMock(return_value=terminal_agent_mock),
        )
        monkeypatch.setattr(
            "semterm.terminal.TerminalOutputParser", terminal_output_parser_mock
        )

        agent = mrkl_agent.original_initialize_agent()

        # Assert that the agent is an instance of a subclass of the Agent class
        assert issubclass(agent.__class__, Agent)

    def test_initialize_executor(self, mrkl_agent, monkeypatch):
        # Mock the objects used by the method
        terminal_agent_executor_mock = MagicMock(spec=TerminalAgentExecutor)

        # Set the MagicMock instances as the attributes of mrkl_agent
        mrkl_agent.terminal_agent = MagicMock(spec=TerminalAgent)
        mrkl_agent.tools = [MagicMock(spec=BaseTool)]
        mrkl_agent.memory = MagicMock(spec=BaseMemory)
        mrkl_agent.max_iterations = 10
        mrkl_agent.verbose = False

        # Mock the constructors and methods
        monkeypatch.setattr(
            "semterm.agent.TerminalAgentExecutor.TerminalAgentExecutor.from_agent_and_tools",
            MagicMock(return_value=terminal_agent_executor_mock),
        )

        executor = mrkl_agent.original_initialize_executor()

        # Assert that the executor is an instance of a subclass of the BaseExecutor class
        assert issubclass(executor.__class__, AgentExecutor)

    def test_run(self, mrkl_agent):
        user_input = "test_input"

        mrkl_agent.terminal_agent_executor.run = MagicMock()
        mrkl_agent.run(user_input)
        mrkl_agent.terminal_agent_executor.run.assert_called_with(input=user_input)
