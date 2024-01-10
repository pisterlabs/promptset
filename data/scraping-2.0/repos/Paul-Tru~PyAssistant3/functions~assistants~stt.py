import whisper
from openai import OpenAI


def whisper_speech_offline(path):
    options = {"language": "de"}
    model = whisper.load_model("small")
    res = model.transcribe(path, **options)
    return res["text"]


def whisper_speech_online(path):
    client = OpenAI()

    audio_file = open(path, "rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
    )
