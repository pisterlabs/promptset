# pylint: disable=redefined-outer-name
from typing import Any

import pydantic
import pytest

from openai_api_wrapper import chat_completion


@pytest.fixture
def response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_message_success(role: str) -> None:
    """Tests the Message class."""
    message = chat_completion.Message(role=role, content="Hello there!")  # type: ignore[arg-type]

    assert message.role == role
    assert message.content == "Hello there!"
    assert str(message) == f"{role}: Hello there!"


def test_message_bad_role() -> None:
    """Tests that an error is raised on a bad role."""
    with pytest.raises(pydantic.ValidationError):
        chat_completion.Message(role="bad", content="Hello there!")  # type: ignore[arg-type]


def test_chat_completion_no_messages_or_prompt() -> None:
    """Tests that an error is raised if neither messages nor a prompt are
    provided.
    """
    with pytest.raises(pydantic.ValidationError):
        chat_completion.ChatCompletion(
            api_key="123", model="gpt-4", system_prompt="", messages=[]
        )


def test_chat_completion_both_messages_and_prompt() -> None:
    """Tests that an error is raised if both messages and a prompt are
    provided.
    """
    with pytest.raises(pydantic.ValidationError):
        chat_completion.ChatCompletion(
            api_key="123",
            model="gpt-4",
            system_prompt="Hello there!",
            messages=[chat_completion.Message(role="user", content="Hi!")],
        )


def test_chat_completion_add_message() -> None:
    """Tests that a message can be added to the chat completion."""
    messages = [chat_completion.Message(role="system", content="Hi!")]
    chat = chat_completion.ChatCompletion(
        api_key="123", model="gpt-4", messages=messages
    )
    new_message = chat_completion.Message(role="user", content="How are you?")
    expected = messages + [new_message]

    chat.add_message(role=new_message.role, content=new_message.content)  # type: ignore[arg-type]

    assert chat.messages == expected


def test_chat_completion_prompt(
    mocker, response: dict[str, str | int | list[dict[str, str | int]]]
) -> None:
    """Tests that a prompt can be run."""
    mocker.patch("openai.ChatCompletion.create", return_value=response)
    chat = chat_completion.ChatCompletion(
        api_key="123",
        model="gpt-4",
        system_prompt="Hello there!",
    )
    chat.add_message(role="user", content="Hi!")
    expected = response["choices"][0]["message"]["content"]  # type: ignore[index]

    actual = chat.prompt()

    assert actual == expected
