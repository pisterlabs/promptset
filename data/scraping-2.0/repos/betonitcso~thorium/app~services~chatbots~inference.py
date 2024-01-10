import logging

import openai
from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST

from app.core.config import config
from app.schemas.chatbots.chat_completion import OpenAIChatCompletionResponse, OpenAIChatCompletionInput
from app.schemas.chatbots.chat_inference import ChatInferenceBody, ChatHistoryItem
from app.services.semantic_search import search


class ChatCompletion:
    def __init__(self):
        openai.api_key = config.openai_api_key
        self.temperature = 0.1

    def get_response(self, language_model: str, payload) -> str:
        try:
            response_raw = openai.ChatCompletion.create(
                model=language_model,
                temperature=self.temperature,
                messages=payload
            )

            response = OpenAIChatCompletionResponse(**response_raw)
            content = response.choices[0].message.content
            return content

        except Exception as e:
            logging.error(e)


chat_completion = ChatCompletion()


def run(body: ChatInferenceBody) -> str:

    context = body.context

    if body.semantic_search:
        if body.semantic_search.limit < 1:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Search limit cannot be smaller than 1.")

        search_results = search.query(body.message, body.semantic_search.tenant, body.semantic_search.limit)
        context += "\n".join(result for result in search_results)

    instructions = get_system_instructions(body.instructions, context)
    openai_input = get_openai_input(instructions, body.message, body.history)

    response = chat_completion.get_response(body.language_model, openai_input)

    return response


def get_openai_input(system_input: OpenAIChatCompletionInput, user_message: str,
                     history: list[ChatHistoryItem]) -> list[dict]:
    chat_history = [
        OpenAIChatCompletionInput(role="user" if item.isUser else "assistant", content=item.text).dict()
        for item in history
    ]

    return [system_input.dict()] + chat_history + [OpenAIChatCompletionInput(role="user", content=user_message).dict()]


def get_system_instructions(instructions: str, context: str) -> OpenAIChatCompletionInput:
    # Since System messages are ignored more frequently, the initial instructions are in user mode.
    system_instruction = OpenAIChatCompletionInput(
        role="user",
        content=f"""
            Instructions: {instructions}
            ---
            Context: {context}
            ---
            """
    )

    return system_instruction
