# https://platform.openai.com/docs/guides/text-to-speech

from openai import OpenAI
import config

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
)


def text_to_speech(text: str, output_filename: str) -> str:
    response = client.audio.speech.create(input=text, voice="alloy", model="tts-1")

    output_filepath = config.tts_base_path + output_filename + ".mp3"
    response.stream_to_file(output_filepath)

    return output_filepath
