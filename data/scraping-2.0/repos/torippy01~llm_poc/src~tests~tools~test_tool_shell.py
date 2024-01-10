import langchain.tools
import pytest

from xecretary_core.tools.tool_shell import ShellAndSummarizeTool


@pytest.fixture
def shell_tool_run(mocker):
    return mocker.patch.object(langchain.tools.ShellTool, "_run")


def test_shell_and_summariza_tool(shell_tool_run):
    tool = ShellAndSummarizeTool()
    tool._run("Hello")
    shell_tool_run.assert_called_once()
