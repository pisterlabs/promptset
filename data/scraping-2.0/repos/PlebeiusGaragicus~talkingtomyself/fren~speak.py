import os
from pathlib import Path
from openai import OpenAI


def nothing(input: str):
    print(f"Speaking: {input}")

    HERE = Path(__file__).parent

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # print(client.models.list())

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=f"{input}"
    )

    speech_file_path = HERE / "speech.mp3"
    response.stream_to_file(speech_file_path)

    # play the file with afplay
    os.system(f"afplay {speech_file_path}")
