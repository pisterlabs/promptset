"""agentml/agents/vision.py"""

from uuid import UUID

from agentml.models import LlmMessage, LlmRole
from agentml.oai import client as openai
from agentml.sandbox import Sandbox

from .base import Agent


class Vision(Agent):
    """Vision Agent"""

    DEFAULT_MODEL = "gpt-4-vision-preview"

    DEFAULT_SYSTEM_MESSAGE = """You are a vision AI assistant.
You are given a task to analyze images and provide insights.

You must use the OpenAI Vision API to analyze the images.
Return a description and any notable findings about the images.
The analysis must be thorough, complete and provide valuable insights.

This analysis will be used to plan the next steps to solve the problem.
"""

    def __init__(
        self,
        session_id: UUID,
        objective: str,
        messages: list[LlmMessage] = None,
        prompt: str = DEFAULT_SYSTEM_MESSAGE,
    ) -> None:
        """
        Vision Agent constructor

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
            ]
        )

        self.analysis: str | None = None

        self._last_messages = []

    def run(self) -> list[LlmMessage]:
        """Run the agent"""
        encoded_images = self.sandbox.get_images_encoded()
        image_messages = [
            {"type": "image_url", "image_url": {"url": image}}
            for image in encoded_images
        ]

        print("Vision.run: Sending request to OpenAI API with images")
        response = openai.chat.completions.create(
            model=self.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": image_messages},
            ],
        )

        print(f"Vision.run: Received response from OpenAI API: {response}")
        self.analysis = response.choices[0].message.content

        messages = [
            LlmMessage(role=LlmRole.USER, content=self.objective),
            LlmMessage(
                role=LlmRole.ASSISTANT,
                content=f"Image analysis:\n{self.analysis}",
            ),
        ]

        self._last_messages = messages

        return messages

    def retry(self) -> list[LlmMessage]:
        """Retry the agent"""
        self.messages.extend(self._last_messages)
        self.messages.append(
            LlmMessage(role=LlmRole.SYSTEM, content="Please try again.")
        )
        return self.run()
