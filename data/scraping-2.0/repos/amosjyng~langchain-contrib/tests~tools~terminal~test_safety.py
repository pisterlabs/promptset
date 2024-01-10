"""Tests for safe terminal execution."""
from unittest.mock import patch

import pytest

from langchain_contrib.llms import BaseHuman
from langchain_contrib.llms.testing import FakeLLM
from langchain_contrib.tools import SafeTerminalChain
from langchain_contrib.utils import temporary_file

vcr = pytest.importorskip("vcr_langchain")


@vcr.use_cassette()
def test_proceed() -> None:
    """Test the human allowing execution as usual."""
    with patch("builtins.input", return_value="Proceed"):
        llm = FakeLLM(sequenced_responses=["date"])
        safe_terminal = SafeTerminalChain(human=BaseHuman())
        assert safe_terminal(llm("Execute a command")) == {
            "choice": "Proceed",
            "command": "date",
            "output": "Fri Mar 24 16:26:49 AEDT 2023",
        }


@vcr.use_cassette()
def test_edit() -> None:
    """Test the human editing the command before execution."""
    human_responses = [
        "Edit command",
        "touch important.txt",
    ]
    # don't check creation because it won't happen apart from the initial run
    with temporary_file("important.txt", check_creation=False):
        with patch("builtins.input", side_effect=human_responses):
            llm = FakeLLM(sequenced_responses=["rm important.txt"])
            safe_terminal = SafeTerminalChain(human=BaseHuman())
            assert safe_terminal(llm("Execute a command")) == {
                "choice": "Edit command",
                "command": "touch important.txt",
                "output": "",
            }
