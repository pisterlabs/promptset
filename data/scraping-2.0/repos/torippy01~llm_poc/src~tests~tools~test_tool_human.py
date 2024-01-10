from langchain.tools.human.tool import HumanInputRun

from xecretary_core.tools.tool_human import create_HumanTool


def test_create_human_tool():
    assert isinstance(create_HumanTool(), HumanInputRun)


def test_get_inpput(mocker):
    tool: HumanInputRun = create_HumanTool()
    get_input = tool.input_func

    input_mock = mocker.patch("builtins.input")

    # in case, press `q` after message
    input_mock.side_effect = ["Hello", "q"]
    expect = get_input()
    assert expect == "Hello"

    # in case, raise EOFError
    input_mock.side_effect = EOFError
    expect = get_input()
    assert expect == ""
