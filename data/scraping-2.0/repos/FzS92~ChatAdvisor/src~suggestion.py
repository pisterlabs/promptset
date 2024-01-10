"""
Process chat history and generate suggested answers using the OpenAI API

Parameters:
- openaikey (str): The API key for accessing the OpenAI API.
- history (str): The chat history string.
- your_usern (str): The username of the user making the request.
- other_usern (str): The username of the other user in the chat.

Returns:
Tuple[str, str, str]: A tuple containing the updated chat history,
the long suggested answer, and the short suggested answer.
"""

from typing import Tuple

import openai

from .utils import (
    add_name,
    remove_bracketed_text,
    remove_extra_newlines,
    replace_comma_before_newline,
    string_to_list,
)


def suggest_answer(
    openaikey: str, history: str, your_usern: str, other_usern: str
) -> Tuple[str, str, str]:
    """Generates a suggested answer based on the given chat history
    and user information using the OpenAI API."""
    openai.api_key = openaikey
    updated_history = history
    updated_history = remove_extra_newlines(updated_history)
    updated_history = remove_bracketed_text(updated_history)
    updated_history = replace_comma_before_newline(updated_history)
    updated_history = add_name(updated_history, your_usern)
    updated_history = (
        "I am a chat assistant and will suggest an answer for the last user.\n"
        + updated_history
    )
    other_usern = string_to_list(other_usern)

    updated_history = str(updated_history)
    other_usernames = str(other_usern)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=updated_history,
        temperature=0.5,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=other_usernames,
    )

    long_answer = response["choices"][0]["text"]

    response = openai.Completion.create(
        model="text-davinci-003",
        # prompt=f"{long_answer}\n\nTl;dr",
        prompt=f"Make my answer shorter: {long_answer}",
        temperature=1,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1,
    )

    short_answer = response["choices"][0]["text"]
    short_answer = remove_extra_newlines(short_answer)
    return updated_history, long_answer, short_answer
