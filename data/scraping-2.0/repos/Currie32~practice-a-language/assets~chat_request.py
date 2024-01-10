import os
import re
import time
from typing import Dict, List

import openai
import requests
from dash import Input, Output, callback, no_update
from tenacity import retry, stop_after_attempt, wait_random_exponential

openai.api_key = os.environ.get("OPENAI_KEY")


@callback(
    Output("user-response-text", "value", allow_duplicate=True),
    Output("loading", "style", allow_duplicate=True),
    Output("check-for-audio-file", "data", allow_duplicate=True),
    Input("check-for-audio-file", "data"),
    prevent_initial_call=True,
)
def convert_audio_recording_to_text(check_for_audio_file: bool) -> str:
    """
    Convert the audio recording from the user into text using OpenAI's
    Whisper-1 model.

    Params:
        check_for_audio_file: Whether to check for the audio recording file.

    Returns:
        The text of the user's audio recording.
        The style of the loading icons.
        Stop checking for the user's audio recording.
    """

    audio_recording = "audio_recording.wav"

    while check_for_audio_file:
        if os.path.exists(audio_recording):

            audio_file = open(audio_recording, "rb")
            os.remove(audio_recording)
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            message_user = transcript.to_dict()["text"]

            return message_user, {"display": "none"}, False

        # Wait 0.1 seconds before looking for the audio file again
        time.sleep(0.1)

    return no_update


def get_assistant_message(messages: List[Dict[str, str]]) -> str:
    """
    Get and process the assistant's (OpenAI's model) message to continue the conversation.

    Params:
        messages: The conversation history between the user and the chat model.

    Returns:
        The message from the assistant.
    """

    chat_response = _chat_completion_request(messages)
    message_assistant = chat_response.json()["choices"][0]["message"]["content"]

    # Remove space before "!" or "?"
    message_assistant = re.sub(r"\s+([!?])", r"\1", message_assistant)

    return message_assistant


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def _chat_completion_request(messages: List[Dict[str, str]]) -> Dict:
    """
    Request a response to the user's statement from one of OpenAI's chat models.

    Params:
        messages: The conversation history between the user and the chat model.

    Returns:
        A response from OpenAI's model to the user's statement.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {
        "model": "gpt-3.5-turbo-0613",
        "messages": messages,
        "temperature": 1.5,  # Higher values provide more varied responses
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        return e


def system_content(
    conversation_setting: str,
    language_learn: str,
    language_known: str,
) -> str:
    """
    Write the content message for the system as part of call OpenAI's chat completion API.
    This provide OpenAI's model with some context about the conversation.

    Params:
        conversation_setting: The setting of the conversation between the user and OpenAI's model.
        language_learn: The language that the user wants to learn.
        language_known: The language that the user speaks.

    Returns:
        The content message for the system.
    """

    content = f"Start a conversation about {conversation_setting} in {language_learn}. \
        Provide one statement in {language_learn}, then wait for my response. \
        Do not write in {language_known}. \
        Always finish your response with a question. \
        Example response: Bonjour, qu'est-ce que je peux vous servir aujourd'hui?"

    content = re.sub(r"\s+", " ", content)

    return content
