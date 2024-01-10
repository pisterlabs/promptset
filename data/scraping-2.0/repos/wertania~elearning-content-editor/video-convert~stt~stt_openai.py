from openai import OpenAI
import config
from video_types import Transcript

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
)


def speech_to_text(audio_file_path: str) -> Transcript:
    raise Exception(
        "OpenAI Whisper is currently not supported as it does not export per-sentence timestamps."
    )

    audio_file = open(config.audio_base_path + audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    # return Transcript.from_json()
    # return Transcript([...])
