import re
from io import BytesIO
from loguru import logger
from typing import Type

from langchain.agents import BaseSingleActionAgent, AgentExecutor
from langchain.tools import BaseTool

from codeinterpreter.chains import get_file_modifications
from codeinterpreter.agents import CustomOpenAIFunctionsAgent
from codeinterpreter.prompts import system_message
from codeinterpreter.schema import File, AIResponse, UserRequest
from codeinterpreter.custom_llm import CustomChatOpenAI

from codeinterpreter.localbox import (LocalBox, upload, download)

from pydantic import BaseModel, Field


class CodeInterpreterInput(BaseModel):
    code: str = Field(description="The code to execute")


class CodeInterpreterTool(BaseTool):
    name = "python"
    description = """
                Input a string of code to a ipython interpreter.
                Write the entire code in a single string. This string can
                be really long, so you can use the `;` character to split lines. 
                Start your code on the same line as the opening quote.
                Do not start your code with a line break.
                For example, do 'import numpy', not '\\nimport numpy'.
                Variables are preserved between runs. 
                """
    args_schema: Type[BaseModel] = CodeInterpreterInput

    def _run(self, code: str):
        ci = CodeInterpreter()
        run_response = ci.run_handler(code)
        return run_response

    def _arun(self, ticker: str):
        raise NotImplementedError("python run does not support async")


class CodeInterpreter:
    output_files: list[File] = []

    def __init__(self):
        self.codebox = LocalBox()
        self.verbose = True
        self.llm = CustomChatOpenAI()
        self.agent_executor = self.agent_executor()

    def agent_executor(self) -> AgentExecutor:
        return AgentExecutor.from_agent_and_tools(
            agent=self.agent(),
            max_iterations=12,
            tools=[CodeInterpreterTool()],
            verbose=self.verbose,
        )

    def agent(self) -> BaseSingleActionAgent:
        return CustomOpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm,
            tools=[CodeInterpreterTool()],
            system_message=system_message
        )

    def run_handler(self, code: str) -> str:
        logger.info("code: = {}", code)
        output = self.codebox.run(code)
        logger.info("output: = {}", output)
        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                        r"ModuleNotFoundError: No module named '(.*)'",
                        output.content,
                ):
                    self.codebox.install(package.group(1))
                    return (
                        f"{package.group(1)} was missing but"
                        "got installed now. Please try again."
                    )
            else:
                pass

        if modifications := get_file_modifications(code, self.llm):
            for filename in modifications:
                file_out = download(filename)
                if not file_out.content:
                    continue
                file_buffer = BytesIO(file_out.content)
                file_buffer.name = filename
                self.output_files.append(
                    File(name=filename, content=file_buffer.read())
                )
        return output.content

    def generate_response(self, user_msg: str, file: File = None, ):
        user_request = UserRequest(content=user_msg, file=file)
        try:
            self._input_handler(user_request)
            self.codebox.start()
            response = self.agent_executor.run(input=user_request.content)
            self.codebox.stop()
            return self._output_handler(response)
        except Exception as e:
            logger.error("Error in CodeInterpreter: e = {}",e)
            return AIResponse(
                content="Error in CodeInterpreter: "
                        f"{e.__class__.__name__}  - {e}"
            )

    def _input_handler(self, request: UserRequest) -> None:
        if not request.file:
            return
        if not request.content:
            request.content = (
                "I uploaded, just text me back and confirm that you got the file(s)."
            )
        request.content += "\n**The user uploaded the following files: **\n"
        request.content += f"[Attachment: {request.file.name}]\n"
        upload(request.file.name, request.file.content)
        request.content += "**File(s) are now available in the cwd. **\n"

    def _output_handler(self, final_response: str):

        output_files = self.output_files

        return AIResponse(content=final_response, files=output_files)
