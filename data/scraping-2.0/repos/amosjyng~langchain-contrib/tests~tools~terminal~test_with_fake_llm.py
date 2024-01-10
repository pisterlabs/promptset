"""Test agent usage of terminal without vcr_langchain."""

from langchain.agents import initialize_agent

from langchain_contrib.llms.testing import FakeLLM
from langchain_contrib.tools import load_tools
from langchain_contrib.utils import current_directory


def test_use_terminal() -> None:
    """Check that the agent can use the terminal.

    This should expose the terminal's statefulness in a way that the regular
    `BashProcess` does not support.
    """
    with current_directory():
        llm = FakeLLM(
            sequenced_responses=[
                "List folders\nAction: Terminal\nAction Input: ls",
                "Enter folder\nAction: Terminal\nAction Input: cd langchain_contrib",
                "List folders\nAction: Terminal\nAction Input: ls",
                "I now know the final answer\nFinal Answer: some folders",
            ]
        )
        tools = load_tools(["persistent_terminal"], llm=llm)
        agent = initialize_agent(tools, llm, verbose=True)
        result = agent.run(
            "List the folders in the current directory. Enter into one of them. List "
            "folders again."
        )
        assert (
            result == "some folders"
        )  # todo: need some way of checking terminal output
