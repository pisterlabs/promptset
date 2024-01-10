import pytest
from ..parser import CustomOutputParser
from langchain.schema import AgentAction, AgentFinish

# Test cases for different types of outputs
test_cases = [
    (
        "Thought: Some thought text\nAction: None",
        AgentFinish(
            return_values={"output": "Some thought text\n"},
            log="Thought: Some thought text\nAction: None",
        ),
    ),
    (
        "Thought: Some thought text\nFinal Answer: The final answer",
        AgentFinish(
            return_values={"output": "The final answer"},
            log="Thought: Some thought text\nFinal Answer: The final answer",
        ),
    ),
    (
        "Thought: Some thought text\nAction: Tool\nAction Input: Tool input",
        AgentAction(
            tool="Tool",
            tool_input="Tool input",
            log="Thought: Some thought text\nAction: Tool\nAction Input: Tool input",
        ),
    ),
]

@pytest.mark.parametrize("input_text,expected_result", test_cases)
def test_custom_output_parser(input_text, expected_result):
    parser = CustomOutputParser()
    result = parser.parse(input_text)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
