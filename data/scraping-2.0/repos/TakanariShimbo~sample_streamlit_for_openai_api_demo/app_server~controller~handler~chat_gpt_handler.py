from typing import Callable, List, Optional

from openai import OpenAI, Stream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)

from .open_ai_handler import OpenAiHandler


def convert_entity_to_message_param(role: str, content: str) -> ChatCompletionMessageParam:
    if role == "user":
        return ChatCompletionUserMessageParam(role="user", content=content)
    elif role == "assistant":
        return ChatCompletionAssistantMessageParam(role="assistant", content=content)
    elif role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=content)
    else:
        raise ValueError("role is 'user' or 'assistant' or 'system'")


class ChatGptHandler(OpenAiHandler):
    @classmethod
    def query_answer(
        cls,
        client: OpenAI,
        prompt: str,
        assistant_id: str = "gpt-3.5-turbo",
        message_prams: Optional[List[ChatCompletionMessageParam]] = None,
    ) -> str:
        response = client.chat.completions.create(
            model=assistant_id,
            messages=cls.get_message_params_added_prompt(prompt=prompt, message_prams=message_prams),
        )

        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("Response from OpenAI API is empty.")
        return answer

    @classmethod
    def query_streamly_answer_and_display(
        cls,
        client: OpenAI,
        prompt: str,
        assistant_id: str = "gpt-3.5-turbo",
        message_prams: Optional[List[ChatCompletionMessageParam]] = None,
        callback_func: Callable[[str], None] = print,
    ) -> str:
        streamly_answer = cls.query_streamly_answer(client=client, prompt=prompt, assistant_id=assistant_id, message_prams=message_prams)
        answer = cls.display_streamly_answer(streamly_answer=streamly_answer, callback_func=callback_func)
        return answer

    @classmethod
    def query_streamly_answer(
        cls,
        client: OpenAI,
        prompt: str,
        assistant_id: str = "gpt-3.5-turbo",
        message_prams: Optional[List[ChatCompletionMessageParam]] = None,
    ) -> Stream[ChatCompletionChunk]:
        streamly_answer = client.chat.completions.create(
            model=assistant_id,
            messages=cls.get_message_params_added_prompt(prompt=prompt, message_prams=message_prams),
            stream=True,
        )

        return streamly_answer

    @staticmethod
    def display_streamly_answer(
        streamly_answer: Stream[ChatCompletionChunk],
        callback_func: Callable[[str], None] = print,
    ):
        answer = ""
        for chunk in streamly_answer:
            answer_peace = chunk.choices[0].delta.content or ""  # type: ignore
            answer += answer_peace
            callback_func(answer)
        return answer

    @staticmethod
    def get_message_params_added_prompt(prompt: str, message_prams: Optional[List[ChatCompletionMessageParam]]) -> List[ChatCompletionMessageParam]:
        if message_prams == None:
            message_prams = []

        copyed_message_params = message_prams.copy()
        copyed_message_params.append(ChatCompletionUserMessageParam(role="user", content=prompt))
        return copyed_message_params
