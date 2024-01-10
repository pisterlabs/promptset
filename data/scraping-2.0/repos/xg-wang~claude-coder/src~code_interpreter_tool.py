from typing import Optional
from io import BytesIO
import logging
import docker
import subprocess
from docker.errors import BuildError, ContainerError, ImageNotFound, APIError
from docker import DockerClient
from pydantic import Field
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from src.util import setup_logging

def _clean_code(code: str) -> str:
    """Clean code from Markdown code blocks."""
    code = code.strip()
    if code.startswith("```python"):
        code = code.removeprefix("```python")
    elif code.startswith("```py"):
        code = code.removeprefix("```py")
    elif code.startswith("```"):
        code = code.removeprefix("```")
    code = code.removesuffix("```")
    if code.startswith("`"):
        code = code.strip("`")
    code = code.strip()
    return code

class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = "Executes Python code in a Docker container. It returns the log output of the Python code. If the output is stderr, use it to improve your code. Input should be a valid Python program."
    client: DockerClient = Field(default_factory=docker.from_env)

    DOCKERFILE = """
    FROM python:3.10
    RUN pip install numpy pandas matplotlib seaborn pydantic chromadb onnxruntime hupper cowsay
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = docker.from_env()
        logging.info("Building Docker image")
        try:
            self.client.images.build(
                fileobj=BytesIO(self.DOCKERFILE.encode('utf-8')), tag="python_code_interpreter:latest"
            )
        except BuildError as ex:
            logging.exception(f"Docker image build failed: {ex.msg}")
            for line in ex.build_log:
                logging.info(line)

    def _run(
        self,
        code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run code in a Docker container."""
        code = _clean_code(code)
        logging.info(f"Running code\n```\n{code}\n```\n")
        # Use subprocess.run to call docker run, and return the stdio or stderr
        p = subprocess.run(
            ["docker", "run", "--rm", "-i", "python_code_interpreter:latest"],
            input=code,
            capture_output=True,
            check=False,
            encoding='utf-8',
            text=True,
        )
        if p.returncode != 0:
            # logging.info(f"returncode: {p.returncode}")
            # logging.info(f"stdout: {p.stdout}")
            # logging.info(f"stderr: {p.stderr}")
            return p.stderr or "The code is not valid"
        else:
            return p.stdout

    async def _arun(
        self,
        code: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


if __name__ == "__main__":
    setup_logging()
    load_dotenv(find_dotenv())

    tool = CodeInterpreterTool()
    agent = initialize_agent(
        [tool],
        ChatAnthropic(temperature=0, model='claude-2'),
        # ChatOpenAI(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    question = "<Instruction>When you have the answer, always say 'Final Answer:'</Instruction>\nWhat is the result of 10!"
    # agent.run(question)
    for step in agent.iter(question):
        # logging.info(f"Step: {step}")
        if output := step.get("intermediate_step"):
            action, value = output[0]
            logging.info(f"action:\n{action.tool}")
            logging.info(f"tool input:\n{action.tool_input}")
            logging.info(f"value:\n{value}")
        elif output := step.get("output"):
            logging.info(f"Output: {output}")
