from abc import ABC

import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMemory,
    SystemMessage,
)
from langchain.tools import BaseTool
from pydantic.typing import NoneType

from semterm.agent.TerminalAgent import TerminalAgent
from semterm.agent.TerminalAgentExecutor import TerminalAgentExecutor
from semterm.langchain_extensions.schema import AgentMistake


class MockTool(BaseTool, ABC):
    name = "mock_tool"
    description = "Mock tool for testing purposes."

    def _run(self, *args, **kwargs):
        pass

    def _arun(self):
        pass


class TestTerminalAgentExecutor:
    @pytest.fixture
    def executor(self, mock_tools, terminal_agent):
        memory = MagicMock(spec=BaseMemory)
        return TerminalAgentExecutor.from_agent_and_tools(
            terminal_agent,
            [MockTool(name="Tool1"), MockTool(name="Tool2"), MockTool(name="Tool3")],
            max_iterations=10,
            verbose=True,
            memory=memory,
        )

    @patch.object(
        TerminalAgent,
        "plan",
        return_value=AgentFinish(
            return_values={"output": "42"},
            log='{"action": "Final Answer", "action_input": "42"}',
        ),
    )
    def test_take_next_step_returns_finish(self, plan_mock, executor):
        # Test that _take_next_step returns AgentFinish when the output is an instance of AgentFinish
        output = AgentFinish(
            {"output": "42"}, '{"action": "Final Answer", "action_input": "42"}'
        )
        result = executor._take_next_step({}, {}, {}, [])
        assert result == output

    @patch.object(
        TerminalAgent,
        "plan",
        return_value=AgentAction(tool="tool1", tool_input="input1", log="input1"),
    )
    @patch.object(MockTool, "run", return_value="observation1")
    def test_take_next_step_returns_actions(self, run_mock, plan_mock, executor):
        # Test that _take_next_step returns a list of AgentAction and observation tuples
        name_to_tool_map = {"tool1": MockTool()}
        color_mapping = {"tool1": "red"}
        inputs = {"input1": "value1"}
        intermediate_steps = []
        result = executor._take_next_step(
            name_to_tool_map, color_mapping, inputs, intermediate_steps
        )
        assert len(result) == 1
        assert isinstance(result[0][0], AgentAction)
        assert result[0][0].tool == "tool1"
        assert result[0][0].tool_input == "input1"
        assert isinstance(result[0][1], str)
        assert result[0][1] == "observation1"

    @patch.object(
        TerminalAgent,
        "plan",
        return_value=AgentMistake(
            log="Invalid input", tool_input="input1", tool="tool1"
        ),
    )
    def test_take_next_step_returns_mistakes(self, plan_mock, executor):
        # Test that _take_next_step returns a list of AgentMistake and observation tuples
        name_to_tool_map = {"tool1": MockTool()}
        color_mapping = {"tool1": "red"}
        inputs = {"chat_history": [SystemMessage(content="Hello")], "input": "value1"}
        intermediate_steps = []

        result = executor._take_next_step(
            name_to_tool_map,
            color_mapping,
            inputs,
            intermediate_steps,
        )
        assert len(result) == 1
        assert isinstance(result[0][0], AgentMistake)
        assert result[0][0].log == "Invalid input"
        assert result[0][0].tool_input == "input1"
        assert result[0][0].tool == "tool1"
        assert isinstance(result[0][1], NoneType)

    @patch.object(
        TerminalAgent,
        "plan",
        return_value=AgentAction(
            log="Unknown tool", tool_input="input1", tool="unknown_tool"
        ),
    )
    def test_take_next_step_returns_invalid_tool(self, plan_mock, executor):
        # Test that _take_next_step returns a list of AgentMistake and observation tuples
        name_to_tool_map = {"tool1": MockTool()}
        color_mapping = {"tool1": "red"}
        inputs = {
            "chat_history": [SystemMessage(content="Hello")],
            "input": "value1",
        }
        intermediate_steps = []

        result = executor._take_next_step(
            name_to_tool_map,
            color_mapping,
            inputs,
            intermediate_steps,
        )
        assert len(result) == 1
        assert isinstance(result[0][0], AgentAction)
        assert result[0][0].log == "Unknown tool"
        assert result[0][0].tool_input == "input1"
        assert result[0][0].tool == "unknown_tool"
        assert result[0][1] == "unknown_tool is not a valid tool, try another one."

    @patch.object(
        TerminalAgent,
        "plan",
        return_value=AgentAction(log="input1", tool_input="input1", tool="tool1"),
    )
    def test_take_next_step_returns_directly(self, plan_mock, executor):
        name_to_tool_map = {"tool1": MockTool(return_direct=True)}
        color_mapping = {"tool1": "green"}
        inputs = {
            "chat_history": [SystemMessage(content="Hello")],
            "input": "value1",
        }
        intermediate_steps = []

        result = executor._take_next_step(
            name_to_tool_map,
            color_mapping,
            inputs,
            intermediate_steps,
        )
        assert len(result) == 1
        assert isinstance(result[0][0], AgentAction)
        assert result[0][0].log == "input1"
        assert result[0][0].tool_input == "input1"
        assert result[0][0].tool == "tool1"
        assert result[0][1] == None
