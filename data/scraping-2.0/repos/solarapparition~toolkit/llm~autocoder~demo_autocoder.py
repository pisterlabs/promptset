"""Autocode a function from a name and functionality description."""

import os
import sys

sys.path.append("")

from assistants import BasicAssistant
from autocoder.drafting import draft_function


def demo():
    """Demo the function drafting process."""
    from langchain.llms import OpenAI

    # Set up the OpenAI API key in the environment, if needed
    # os.environ["OPENAI_API_KEY"] = "<your key>"

    # If the key is not set, the code will raise an error
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "Please set the OpenAI API key as an environment variable called `OPENAI_API_KEY`."
        )

    # function_name = "fibonacci"
    # functionality = "generates the nth fibonacci number"

    function_name = "get_python_architecture_structure"
    functionality = "Given a directory of Python packages and modules, create a dictionary that describes the structure of the packages, modules, classes, and functions within the directory."

    model_name = "text-davinci-003"
    text_llm = OpenAI(temperature=0, model_name=model_name, max_tokens=-1)
    assistant = BasicAssistant(text_llm)
    results = draft_function(
        name=function_name, functionality=functionality, assistant=assistant
    )
    func_draft = results.code
    print(func_draft)


if __name__ == "__main__":
    demo()
