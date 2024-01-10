import json
import os
import re
from os import path

from openai import AsyncOpenAI

from tools.common import Tool, ToolCallResult
from utils.helpers import get_random_string
from utils.llm import get_response_message_from_gpt
from utils.logger import Logger
import subprocess

separator = "\n`````\n"


class CodeWriter(Tool):
    name: str = "code_writer"
    description: str = "This tool is used to compute or plot graph by creating a python script."

    _directory: str = "./coding"
    _environment: str = "python 3.10"
    _model_name = "gpt-4-1106-preview"
    _session_identifier: str = get_random_string(8)

    async def run(self, title: str, problem: str, conditions: str, arguments: str) -> ToolCallResult:
        Logger.info(f"tool:{self.name}")
        directories = [self._get_directory(),
                       self._get_directory() + "/output"]

        for directory in directories:
            os.mkdir(directory)

        response = await self._generate_script(problem, conditions, arguments)

        self._extract_python_code(response)

        output = self._run_script()

        if output is not None:
            print("Docker command output:")
            print(output)

        result = self._get_final_result()

        return ToolCallResult(result=json.dumps({
            "status": "execution_completed",
            "title": self.title,
            "task_id": self._session_identifier,
            "result": result,
        }))

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_title": {
                            "type": "string",
                            "description": "A title to the task.",
                        },
                        "problem": {
                            "type": "string",
                            "description": "The problem you need to solve by running a script.",
                        },
                        "conditions": {
                            "type": "string",
                            "description": "The conditions of the problem.",
                            "default": ""
                        },
                        "arguments": {
                            "type": "string",
                            "description": "The arguments of the problem.",
                            "default": ""
                        },
                    },
                    "required": ["title", "problem"]
                },

            }
        }

    async def _generate_script(self, problem: str, conditions: str, arguments: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"""You are a software engineer. You are given a requirement and you need to write a python script to solve it.
                If you need to plot a graph, you can use matplotlib and save the graph in ./output directory. 
                The Python version: {self._environment}
                Extra packages: matplotlib, pandas, numpy, yfinance, mplfinance, scikit-learn, pyppeteer, bs4, torch, torchvision, torchaudio, diagrams"""
            },
            {
                "role": "user",
                "content": f"Please write a python script to solve the following problem:\n{problem}\n{separator}Conditions:\n{conditions}\n{separator}\nArguments:\n{arguments}\n{separator}"
            }]
        message = await get_response_message_from_gpt(messages=messages, model_name=self._model_name)

        return message.content

    def _extract_python_code(self, markdown_content):
        """
        Extracts Python code blocks from Markdown content.

        Args:
        markdown_content (str): A string containing Markdown content.

        Returns:
        list: A list of strings, each representing a Python code block.
        """
        pattern = r"```python\n(.*?)\n```"
        code = re.findall(pattern, markdown_content, re.DOTALL)

        file_path = self._get_directory() + "/code.py"

        with open(file_path, "w") as f:
            f.write(code[0])

    def _run_script(self):

        command = f"""docker run --rm -v "$(pwd)/{self._directory}/{self._session_identifier}:/app" -v "$(pwd)/{self._directory}/{self._session_identifier}/output:/app/output" python_runtime python code.py > $(pwd)/{self._directory}/{self._session_identifier}/output/result.log 2>&1"""
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")
            return None

    def _get_result(self) -> str:
        file_path = self._get_directory + f"/output/result.log"
        with open(file_path, "r") as f:
            return f.read()

    async def _get_final_result(self) -> str:
        result = self._get_result()

        has_error = await self._check_is_error(result)

        if has_error:
            response = await self._regenerate_script(result)

            self._extract_python_code(response)

            self._run_script()

            return await self._get_final_result()

        else:
            return result

    async def _check_is_error(self, result: str) -> bool:

        messages = [
            {
                "role": "system",
                "content": f"""You are a software engineer. You need to check the output has any error."""
            },
            {
                "role": "user",
                "content": f"Output:\n{result}"
            }
        ]

        response = await AsyncOpenAI().chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "output_checker",
                    "description": "This tool is used to verify the output has any error.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "has_error": {
                                "type": "boolean",
                                "description": "The output has any error.",
                            },
                        },
                        "required": ["has_error"]
                    },

                }
            }],
            tool_choice={"type": "function", "function": {"name": "output_checker"}}
        )
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return args["has_error"]

    async def _regenerate_script(self, error):

        file_path = self._get_directory() + "/code.py"

        with open(file_path, "r") as f:
            code = f.read()

        messages = [
            {
                "role": "system",
                "content": f"""You are a software engineer. """
            },
            {
                "role": "user",
                "content": f"Code:\n{code}\n{separator}Error:\n{error}\n{separator}"
            }
        ]

        message = await get_response_message_from_gpt(messages=messages, model_name=self._model_name)

        self._extract_python_code(message.content)

        self._run_script()

    def _get_directory(self) -> str:
        return f"{self._directory}/{self._session_identifier}"
