from decouple import config
from typing import Literal, get_args
from pathlib import Path
from openai import OpenAI


client = OpenAI(api_key=config("OPENAI_API_KEY"))
current_directory = Path(__file__).parent
Model = Literal["tts-1", "tts-1-hd"]
Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def text_to_speech(
    input_text: str,
    model: Model,
    voice: Voice,
    file_name: str,
):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input_text,
    )
    response.stream_to_file(f"{current_directory}/{file_name}.mp3")


# for voice in get_args(Voice):
#     text_to_speech(
#         input_text="My name is Spongebob and I love pineapples!",
#         model="tts-1",
#         voice=voice,
#         file_name=voice,
#     )


for model in get_args(Model):
    text_to_speech(
        input_text="My name is Spongebob and I love pineapples!",
        model=model,
        voice="echo",
        file_name=model,
    )
