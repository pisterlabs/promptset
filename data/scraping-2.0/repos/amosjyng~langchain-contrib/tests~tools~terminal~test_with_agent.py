"""Test agent usage of terminal."""

import pytest
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

from langchain_contrib.tools import load_tools
from langchain_contrib.utils import current_directory

vcr = pytest.importorskip("vcr_langchain")


@vcr.use_cassette()
async def test_use_terminal() -> str:
    """Check that the agent can use the terminal.

    This should expose the terminal's statefulness in a way that the regular
    `BashProcess` does not support.
    """
    with current_directory():
        llm = OpenAI(temperature=0)  # type: ignore
        tools = load_tools(["persistent_terminal"], llm=llm)
        agent = initialize_agent(tools, llm, verbose=True)
        result = agent.run(
            "List the folders in the current directory. Enter into one of them. List "
            "folders again."
        )
        assert "The folders in the current directory" in result
        return result


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_use_terminal)
