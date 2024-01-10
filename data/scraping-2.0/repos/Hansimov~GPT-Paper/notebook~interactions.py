from IPython.core.magic import (
    register_line_magic,
    register_cell_magic,
    register_line_cell_magic,
)
from cells import get_above_cell_content
from agents.openai import OpenAIAgent


@register_line_cell_magic
def chat(line=None, cell=None, system_message=None):
    above_cell_content = get_above_cell_content()
    print(f"line: {line}")
    print(f"cell: {cell}")
    print(f"above_content:  {above_cell_content}")
    print(f"system_message: {system_message}")
