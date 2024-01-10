import os
import pickle
from pathlib import Path

import openai


def prompt_gpt(
    model: str = None,
    messages: list[str] = None,
) -> dict:
    """
    Submit a prompt to ChatGPT API including a conversation history.

    Args:
    ---
    model: The name of the model
    messages: A list of messages the includes the prompt and the conversation history

    Returns:
    ---
    A dictionary that has the following keys:
    - msg: ChatGPT response
    - msgs: The updated conversation history
    - input_tokens: Number of input tokens used
    - output_tokens: Number of output tokens used

    Useful links:
    ---
    models: https://platform.openai.com/docs/models/gpt-3-5
    account balance: https://platform.openai.com/account/billing/overview
    create params: https://platform.openai.com/docs/api-reference/chat/create
    pricing: https://openai.com/pricing
    """

    response = openai.ChatCompletion.create(model=model, messages=messages)
    msg = response.get("choices")[0].get("message").get("content")

    assistant = {"role": "assistant", "content": msg}
    messages.append(assistant)

    input_tokens = response.get("usage").get("prompt_tokens")
    output_tokens = response.get("usage").get("prompt_tokens")

    return {
        "msg": msg,
        "msgs": messages,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def create_messages(
    prompt: str = None,
    role: str = None,
    messages: list[dict] = None,
):
    """Adds the user prompt to the conversation history."""

    if prompt is None:
        raise ValueError("prompt cannot be None")

    if messages is None:
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ]
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


def load_conversation(msgs_path: Path, new_conversation: bool):
    conversation_history = None

    if not new_conversation:
        if not os.path.exists(msgs_path):
            print("No history to load")
        else:
            try:
                with open(msgs_path, mode="rb") as f:
                    conversation_history = pickle.load(f)
            except Exception as e:
                print(e)

    return conversation_history
