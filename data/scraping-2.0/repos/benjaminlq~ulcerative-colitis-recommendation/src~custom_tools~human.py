from typing import Callable, Optional

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool

def _print_func(text: str) -> None:
    print("\n")
    print(text)
    
def _get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

class HumanTool(BaseTool):
    """Tool that asks user for input."""

    name = "Human"
    description = (
        "You can ask a human for additional information when you got stuck "
        "or you are not sure about the treatment recommendation. "
        "The input should be a question for the human to provide additional "
        "information about the patient, such as age, gender, whether the patient "
        "under treatment is a new patient or under maintenance, whether the patient "
        "has any special response/failure to any drugs or other conditions such as "
        "pregnancy, extraintestinale manifestations, etc"
    )
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _print_func)
    input_func: Callable = Field(default_factory=lambda: _get_input)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Human input tool."""
        self.prompt_func(query)
        return self.input_func()