"""Test the Chain wrapper for Tools."""
import pytest
from langchain.chains.sequential import SequentialChain

from langchain_contrib.chains.testing import FakeChain
from langchain_contrib.chains.tool import ToolChain
from langchain_contrib.tools import TerminalTool

vcr = pytest.importorskip("vcr_langchain")


def test_chain_can_be_called() -> None:
    """Test that the chain can be called directly."""
    terminal = TerminalTool()
    chain = ToolChain(tool=terminal)
    assert chain({"action_input": 'basename "$PWD"'}, return_only_outputs=True) == {
        "action_result": "langchain-contrib"
    }


@vcr.use_cassette()
def test_chain_can_be_chained() -> None:
    """Test that the chain can be called in a sequential chain."""
    terminal = TerminalTool()
    terminal_chain = ToolChain(tool=terminal, tool_output_key="time")
    sql_chain = FakeChain(
        expected_inputs=["time"],
        output={"events_from_last_24_hrs": "a,b,c"},
    )
    chained = SequentialChain(
        chains=[terminal_chain, sql_chain],
        input_variables=["action_input"],
        output_variables=["time", "events_from_last_24_hrs"],
        return_all=True,
    )
    assert chained({"action_input": "date"}, return_only_outputs=True) == {
        "time": "Thu Mar 23 14:50:01 AEDT 2023",
        "events_from_last_24_hrs": "a,b,c",
    }
