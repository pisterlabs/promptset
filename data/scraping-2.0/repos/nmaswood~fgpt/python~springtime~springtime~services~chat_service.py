import abc
from collections.abc import Generator
from typing import Any

import openai
from loguru import logger

from springtime.models.chat import ChatFileContext, ChatHistory
from springtime.models.open_ai import OpenAIModel
from springtime.services.prompt import create_prompt


class ChatService(abc.ABC):
    @abc.abstractmethod
    def ask_streaming(
        self,
        context: list[ChatFileContext],
        question: str,
        history: list[ChatHistory],
    ) -> Generator[Any, Any, None]:
        pass

    @abc.abstractmethod
    def get_prompt(
        self,
        context: list[ChatFileContext],
        question: str,
        history: list[ChatHistory],
    ) -> str:
        pass

    @abc.abstractmethod
    def get_title(self, question: str, answer: str) -> str:
        pass


SYSTEM_1 = """You are an AI assistant that is an expert financial analyst.
Do not use language or provide opinions or judgment on an investment or financials, but provide objective and factual analysis.
Use the data provided, but analyze it objectively and in a fact-based manner.
Answer all questions accurately, especially when you include data.
If you make a calculation, outline your methodology or assumptions clearly.
Round and use an easy to read numeric format when showing numbers.
Do not use language or provide opinions or judgment on an investment or financials, but provide objective and factual analysis.
"""

SYSTEM_2 = """
Output your response in well formatted markdown.
For example:

* for list responses use bullet points
* for headers use #, ##, ###, etc.
* for links use [link text](link url)
* for tables use table elements
* use br tags for line breaks
"""


class OpenAIChatService(ChatService):
    def __init__(self, model: OpenAIModel) -> None:
        self.model = model

    def get_prompt(
        self,
        context: list[ChatFileContext],
        question: str,
        history: list[ChatHistory],
    ) -> str:
        return f"""
System: {SYSTEM_1}
System: {SYSTEM_2}
User: {create_prompt(context, question, history)}
        """

    def ask_streaming(
        self,
        context: list[ChatFileContext],
        question: str,
        history: list[ChatHistory],
    ) -> Generator[Any, Any, None]:
        prompt = create_prompt(context, question, history)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_1},
                {
                    "role": "system",
                    "content": SYSTEM_2,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=True,
        )
        for resp in response:
            choices = resp["choices"]
            delta = choices[0].get("delta")
            if not delta:
                continue
            content = delta.get("content")
            if content:
                yield content

    def get_title(self, question: str, answer: str) -> Generator[Any, Any, None]:
        prompt = f"""
        Based on the question and answer please respond with a concise, accurate title for the exchange.
        Do not output anything except the title itself. Try to limit your response to at most five words.

        Question: {question}
        Answer: {answer}
        """.format(
            question=question,
            answer=answer,
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst chat bot. The user asked you the following question and you responded with the following answer.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        choices = response["choices"]
        if len(choices) == 0:
            logger.warning("No choices returned from OpenAI")
        first_choice = choices[0]
        return first_choice["message"]["content"]
