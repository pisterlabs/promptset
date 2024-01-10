from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.utilities.python import PythonREPL

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class CustomPythonREPLSchema(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    query: str = Field(description="The code to be executed.")
    output_file: str = Field(description="The final csv file you store in codes.")


class CustomPythonREPLTool(PythonREPLTool):
    """A tool for running python code in a REPL."""

    name: str = "MobilityGPT_Python_REPL"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
        "Pass the output_file argument to define the final csv file you store in codes."
    )
    args_schema: Type[CustomPythonREPLSchema] = CustomPythonREPLSchema

    def _run(
        self,
        query: str,
        output_file: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        return super()._run(query, run_manager)

    async def _arun(
        self,
        query: str,
        output_file: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        return super()._arun(query, run_manager)