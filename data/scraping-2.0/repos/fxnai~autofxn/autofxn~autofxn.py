# 
#   Function
#   Copyright © 2023 NatML Inc. All Rights Reserved.
#

from json import loads
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import openai
from pydantic import BaseModel, Field
from typing import List

class PredictorNotebook (BaseModel):
    source: str = Field(description="Raw Python code to be executed.")
    pip_packages: List[str] = Field(description="Python package dependencies to install with the `pip` package manager.")
    system_packages: List[str] = Field(description="System package dependencies to install with the `apt-get` package manager.")

DIRECTIVES = [
    "Your response must contain a Python function called `predict` that conforms to what the user requests.",
    "The `predict` function must have type annotations for its input arguments whenever possible.",
    "If your code requires Python dependencies, add an IPython magic line that uses `%pip install` to install Python dependencies.",
    "If your code requires system package dependencies, add an IPython system command line that uses `!apt-get install -y` to install Linux packages.",
    "For input images to the predictor, the function should accept a corresponding Pillow `Image.Image` instance.",
    "For input tensors to the predictor, the function should use a numpy `ndarray` instance instead of a `torch.Tensor`.",
    "For predictors that need to install OpenCV, always install `opencv-python-headless` instead of `opencv-python`",
    "Always add a Google-style docstring to the predictor with a description of the function, its arguments, and what it returns.",
    "Prefer installing dependencies from the Python package manager `pip` instead of the system package manager `apt-get`.",
]

def create_predictor_notebook (
    prompt: str,
    openai_api_key: str=None
) -> NotebookNode:
    """
    Create a predictor notebook given a description of what the predictor does.

    Parameters:
        prompt (str): Description of what the predictor does.
        openai_api_key (str): OpenAI API key. This can also be specified with the `OPENAI_API_KEY` env.

    Returns:
        NotebookNode: Jupyter notebook node.
    """
    # Configure OpenAI
    if openai_api_key:
        openai.api_key = openai_api_key
    # Generate function call schema
    call_name = "generate_predictor"
    call_schema = PredictorNotebook.model_json_schema()
    call_schema.pop("title")
    # Generate source code with AI
    directives = "\n\n".join(DIRECTIVES)
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"""You are an assistant that writes AI prediction functions, or "predictors" for short, given a description of what the function should do.
                {directives}
                """
            },
            { "role": "user", "content": prompt }
        ],
        functions=[
            {
                "name": call_name,
                "description": "A function which generates a Python script that can be executed.",
                "parameters": call_schema
            }
        ],
        function_call={ "name": call_name },
        temperature=0.
    )
    # Parse notebook
    message = completion["choices"][0]["message"]
    call_args = message["function_call"]["arguments"]
    call_args = loads(call_args, strict=False)
    notebook = PredictorNotebook(**call_args)
    # Create predictor card cell
    cells = []
    predictor_card_cell = new_markdown_cell(f"Created by autofxn:\n> {prompt}")
    cells.append(predictor_card_cell)
    # Create system package cell
    if len(notebook.system_packages) > 0:
        system_deps_cell = new_code_cell("!apt-get install -y " + " ".join(notebook.system_packages))
        cells.append(system_deps_cell)
    # Create Python package cell
    if len(notebook.pip_packages) > 0:
        python_deps_cell = new_code_cell("%pip install " + " ".join(notebook.pip_packages))
        cells.append(python_deps_cell)
    # Create source cell
    source_cell = new_code_cell(notebook.source)
    cells.append(source_cell)
    # Create predictor notebook
    notebook = new_notebook()
    notebook["cells"] = cells
    # Return
    return notebook