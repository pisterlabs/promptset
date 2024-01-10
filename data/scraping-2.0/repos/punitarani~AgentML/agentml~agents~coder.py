"""agentml/agents/coder.py"""

import re
from uuid import UUID

from agentml.models import LlmMessage, LlmRole
from agentml.oai import client as openai
from agentml.sandbox import Sandbox

from .base import Agent


class Coder(Agent):
    """Coder Agent"""

    DEFAULT_MODEL = "gpt-4-1106-preview"

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant that writes python code.
You are given a task to solve along with the context of the previous steps.

You are only allowed to use python code to solve the task.
The code block must be a valid starting with ```python and ending with ```.

There must only be 1 code block in the response.
The code block must include the entire code to solve the task.
Do not suggest incomplete code which requires users to modify.

All the packages and libraries are already installed.
Only provide the code for main.py.
The dataset is in file "data.csv"

If the code will output a file or image, save the file in the output directory.
This applies to any plots, charts, graphs, or images. Use appropriate name and extensions.
All images must be saved as .jpg files.

Use appropriate variable names and comments to make the code readable.
Use appropriate colors and labels for plots, charts, and graphs.

Rewrite the entire file starting from the imports and ending,
with the last line outputting the result of the intended task.
    """

    def __init__(
        self,
        session_id: UUID,
        objective: str,
        messages: list[LlmMessage] = None,
        prompt: str = DEFAULT_SYSTEM_MESSAGE,
    ) -> None:
        """
        Coder Agent constructor

        Args:
            session_id (UUID): Session ID
            objective (str): Objective of the agent
            messages (list[LlmMessage], optional): List of messages to be used for the agent. Defaults to [].
            prompt (str, optional): Prompt to be used for the agent. Defaults to DEFAULT_SYSTEM_MESSAGE.
        """

        super().__init__(
            session_id=session_id, objective=objective, messages=messages, prompt=prompt
        )

        self.sandbox = Sandbox(session_id=session_id)

        self.messages.extend(
            [
                LlmMessage(role=LlmRole.SYSTEM, content=self.prompt),
                LlmMessage(role=LlmRole.USER, content=self.objective),
                LlmMessage(role=LlmRole.USER, content=self.sandbox.get_file_content()),
            ]
        )

        self._last_messages = None

        self.code: str | None = None

    def run(self) -> list[LlmMessage]:
        """Run the agent"""
        print(f"Coder.run: Sending request to OpenAI API: {self.objective}")
        response = openai.chat.completions.create(
            model=self.DEFAULT_MODEL,
            messages=self.get_messages(),
        )

        print(f"Coder.run: Received response from OpenAI API: {response}")
        response_content = response.choices[0].message.content
        matched = re.search(r"```python(.*?)```", response_content, re.DOTALL)
        if matched:
            code = matched.group(1).strip()
        else:
            code = None

        self.sandbox.update(code=code)
        output, output_files = self.sandbox.execute()
        # TODO: validate output

        print(f"Coder.run: Sandbox output: {output}")
        for file in output_files:
            print(f"Coder.run: Sandbox output file: {file}")

        messages = [
            LlmMessage(role=LlmRole.USER, content=self.objective),
            LlmMessage(
                role=LlmRole.ASSISTANT,
                content=f"Here is the code:\n```python\n{code}\n```",
            ),
        ]

        output = self.get_pretty_output(messages, output)

        if output:
            messages.append(
                LlmMessage(
                    role=LlmRole.ASSISTANT, content=f"Here is the output:\n{output}"
                ),
            )

        self._last_messages = messages

        return messages

    def retry(self) -> list[LlmMessage]:
        """Retry the agent"""
        self.messages.extend(self._last_messages)
        self.messages.append(
            LlmMessage(role=LlmRole.SYSTEM, content="Please try again.")
        )
        return self.run()

    @staticmethod
    def get_pretty_output(messages: list[LlmMessage], output: str) -> str:
        """
        Get pretty output from messages

        Args:
            messages (list[LlmMessage]): LLM Messages
            output (str): Output from sandbox
        """

        formatter_prompt = """You are a raw text to pretty markdown converter.
Format the output from the code execution to pretty markdown.
It must be in a markdown as format: ```markdown\n{pretty output}\n```
        """

        # Build messages to send to OpenAI API
        messages = [
            LlmMessage(role=LlmRole.SYSTEM, content=formatter_prompt),
            LlmMessage(role=LlmRole.USER, content=output),
        ]

        # Convert messages to JSON
        messages = [msg.model_dump(mode="json") for msg in messages]

        print("Coder.get_pretty_output: Getting pretty output")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )
        return response.choices[0].message.content
