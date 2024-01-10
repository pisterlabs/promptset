from typing import Callable, List, Optional
from nextcord import Interaction
import openai
from noncommands.chatsplit import chatsplit

# Type aliases for readability
SendMethod = Callable[[str], None]


async def dadroid_single(
    personality: str,
    prompt: str,
    first_send_method: SendMethod,
    send_method: Optional[SendMethod] = None,
    response_starter: str = "",
    chats: List[dict] = [],
    beef: bool = False,
) -> None:
    """Handles single message interaction with the chat model."""
    messages = (
        chats
        if chats
        else [
            {"role": "system", "content": personality},
            {"role": "user", "content": prompt},
        ]
    )

    chat_completion = create_chat_completion(messages, beef=beef)
    await respond_from_chat_completion(
        chat_completion, first_send_method, send_method, response_starter
    )


async def dadroid_response(
    personality: str,
    response: str,
    chats: List[dict] = [],
    beef: bool = False,
) -> None:
    """Handles single message interaction with the chat model."""
    messages = (
        chats
        if chats
        else [
            {"role": "system", "content": personality},
            {"role": "user", "content": response},
        ]
    )

    chat_completion = create_chat_completion(messages, beef=beef)

    return chat_completion.choices[0].message.content


async def dadroid_multiple(
    personality: str,
    messages: List[dict],
    first_send_method: SendMethod,
    send_method: SendMethod,
    beef: bool = False,
) -> None:
    """Handles multiple messages interaction with the chat model."""
    model = "gpt-4-vision-preview" if beef else "gpt-3.5-turbo-1106"
    messages_with_personality = [{"role": "system", "content": personality}] + messages

    try:
        chat_completion = create_chat_completion(messages_with_personality, model)
        await respond_from_chat_completion(
            chat_completion, first_send_method, send_method
        )
    except openai.error.APIError:
        await first_send_method(
            "I'm sorry, my system is currently having some issues. "
            "Send another message! If that doesn't work, wait a few minutes and try again."
        )


def create_chat_completion(
    messages: List[dict], model: str = "gpt-3.5-turbo-1106", beef: bool = False
) -> dict:
    """Creates a chat completion using OpenAI's API."""
    if beef:
        model = "gpt-4-1106-preview"

    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=4000,
    )


async def respond_from_chat_completion(
    chat_completion: dict,
    first_send_method: SendMethod,
    send_method: Optional[SendMethod],
    initial_response: str = "",
) -> None:
    """Sends response from chat completion."""
    send_method = send_method or first_send_method
    messages = chatsplit(initial_response + chat_completion.choices[0].message.content)

    for index, message in enumerate(messages):
        await (first_send_method if index == 0 else send_method)(message)
