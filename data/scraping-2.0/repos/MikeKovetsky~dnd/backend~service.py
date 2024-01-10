import os

from openai import OpenAI
from openai._base_client import HttpxBinaryResponseContent
from openai.types.beta.threads import ThreadMessage
from retrying import retry

from backend.models import Message

client = OpenAI(api_key=os.environ["OPENAI_KEY"])
img_cache = {}


@retry(stop_max_attempt_number=5)
def get_messages(thread_id: str) -> list[Message]:
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    prepared_messages = []
    for i, message in enumerate(messages.data):
        prepared_messages.append(_build_message(message, i == 0))
    prepared_messages.reverse()
    return prepared_messages


def _build_message(message: ThreadMessage, is_last_message: bool) -> Message:
    text = message.content[0].text.value
    if "IMAGE GENERATION PROMPT:" in text and is_last_message:
        if message.id in img_cache:
            return Message(text=img_cache[message.id], role=message.role, type="image")
        else:
            image_url = get_image(text.replace("IMAGE GENERATION PROMPT:", ""))
            img_cache[message.id] = image_url
        return Message(text=image_url, role=message.role, type="image")
    else:
        return Message(text=text, role=message.role, type="text")


@retry(stop_max_attempt_number=5)
def create_thread() -> str:
    thread = client.beta.threads.create()
    return thread.id


@retry(stop_max_attempt_number=5)
def get_voice(text: str) -> HttpxBinaryResponseContent:
    return client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text
    )


@retry(stop_max_attempt_number=5)
def reply(thread_id: str, text: str) -> None:
    client.beta.threads.messages.create(
        thread_id=thread_id,
        content=text,
        role="user",
    )
    old_messages = get_messages(thread_id)
    client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id="asst_RrdkHpZIEPZ9bgwSE6zMGap2",
    )
    while True:
        new_messages = get_messages(thread_id)
        if len(old_messages) != len(new_messages):
            if new_messages[-1].text != "":
                break


@retry(stop_max_attempt_number=5)
def get_image(prompt: str) -> str:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url
