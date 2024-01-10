import glob
import json
import importlib
import inspect
from logging import getLogger
from pathlib import Path
import os
import warnings
from typing import Callable, List, Optional
from openai.types.chat import ChatCompletionMessageToolCall

from democratic_agent.architecture.helpers.tool import Tool
from democratic_agent.chat.chat import Chat

# TODO: Create our own logger.
LOG = getLogger(__name__)


class ToolsManager:
    def __init__(self):
        self.module_path = "democratic_agent.tools.tools"
        self.tools_folder = Path(__file__).parent / "tools"
        self.default_tools = []
        # Ideally the data retrieved after executing tool should be send online to our database (after filtering), for future fine-tuning, so we can improve the models and provide them back to the community.

    def save_tool(self, function, name):
        path = os.path.join(self.tools_folder / f"{name}.py")
        with open(path, "w") as f:
            f.write(function)

    def get_tool(self, name: str) -> Callable:
        with warnings.catch_warnings():
            # Filter ResourceWarnings to ignore unclosed file objects
            warnings.filterwarnings("ignore", category=ResourceWarning)

            # Dynamically import the module
            module = importlib.import_module(f"{self.module_path}.{name}")

        # Retrieve the function with the same name as the module
        tool_function = getattr(module, name, None)

        if tool_function is None:
            raise AttributeError(f"No function named '{name}' found in module '{name}'")

        return tool_function

    def get_all_tools(self) -> List[str]:
        module_names = []

        # Use glob to find all Python files in the specified folder
        python_files = glob.glob(os.path.join(self.tools_folder, "*.py"))
        python_files = [file for file in python_files if "__init__" not in file]

        for file_path in python_files:
            # Remove the file extension and path to get the module name
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            module_names.append(module_name)

        return module_names

    def execute_tools(
        self,
        tools_call: List[ChatCompletionMessageToolCall],
        functions: List[Callable],
        chat: Optional[Chat] = None,
    ) -> List[Tool]:
        functions_dict = {}
        for function in functions:
            functions_dict[function.__name__] = function

        tools_result: List[Tool] = []
        for tool_call in tools_call:
            try:
                function_name = tool_call.function.name
                function = functions_dict[function_name]
            except KeyError:
                if chat:
                    chat.add_tool_feedback(
                        id=tool_call.id,
                        message="Function name doesn't exist, if you want to use a new tool first select it please.",
                    )
                    print(f"Function name: {function_name} doesn't exist...")
                    continue

            signature = inspect.signature(function)
            args = [param.name for param in signature.parameters.values()]

            arguments = json.loads(tool_call.function.arguments)
            call_arguments_dict = {}
            for arg in args:
                # Check if the argument has a default value
                default_value = signature.parameters[arg].default
                arg_value = arguments.get(arg, None)
                if arg_value is None and default_value is inspect.Parameter.empty:
                    raise Exception(
                        f"Function {function_name} requires argument {arg} but it is not provided."
                    )
                # Use the provided value or the default value
                call_arguments_dict[arg] = (
                    arg_value if arg_value is not None else default_value
                )
            try:
                response = function(**call_arguments_dict)
                args_string = ", ".join(
                    [f"{key}={value!r}" for key, value in call_arguments_dict.items()]
                )
                print(f"{function.__name__}({args_string}): {response}")
            except Exception as e:
                response = f"Error while executing function {function_name} with arguments {call_arguments_dict}. Error: {e}"
            if chat:
                chat.add_tool_feedback(id=tool_call.id, message=response)
            tools_result.append(Tool(name=function_name, feedback=response))
        return tools_result
