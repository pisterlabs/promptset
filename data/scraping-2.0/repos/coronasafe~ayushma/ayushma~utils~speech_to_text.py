import os

import openai
import requests
from django.conf import settings
from google.cloud import speech

from ayushma.models.enums import STTEngine


class WhisperEngine:
    def __init__(self, api_key, language_code):
        self.api_key = api_key
        self.language_code = language_code

    def recognize(self, audio):
        # workaround for setting api version ( https://github.com/openai/openai-python/pull/491 )
        current_api_version = openai.api_version
        openai.api_version = "2020-11-07"
        transcription = openai.Audio.transcribe(
            "whisper-1",
            file=audio,
            language=self.language_code.replace("-IN", ""),
            api_key=self.api_key,
            api_base="https://api.openai.com/v1",
            api_type="open_ai",
            api_version="2020-11-07",  # Bug in openai package, this parameter is ignored
        )
        openai.api_version = current_api_version
        return transcription.text


class GoogleEngine:
    def __init__(self, api_key, language_code):
        self.api_key = api_key
        self.language_code = language_code

    def recognize(self, audio):
        client = speech.SpeechClient()
        audio_content = audio.file.read()
        audio_data = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            language_code=self.language_code,
        )

        response = client.recognize(config=config, audio=audio_data)
        if not response.results:
            return ""
        return response.results[0].alternatives[0].transcript


class SelfHostedEngine:
    def __init__(self, api_key, language_code):
        self.language_code = language_code

    def recognize(self, audio):
        response = requests.post(
            settings.SELF_HOSTED_ENDPOINT,
            files={"audio": audio},
            data={
                # change this model to get faster results see: https://github.com/coronasafe/care-whisper
                "model": "small",
                "language": self.language_code.replace("-IN", ""),
            },
        )

        if not response.ok:
            print("Failed to recognize speech with self hosted engine")
            return ""
        response = response.json()
        return response["data"]["transcription"].strip()


engines = {
    "whisper": WhisperEngine,
    "google": GoogleEngine,
    "self_hosted": SelfHostedEngine,
    # Add new engines here
}


def speech_to_text(engine_id, audio, language_code):
    api_key = os.environ.get("STT_API_KEY", "")
    engine_name = STTEngine(engine_id).name.lower()
    engine_class = engines.get(engine_name)

    if not engine_class:
        raise ValueError(f"Invalid STT engine ID: {engine_id}")

    try:
        engine = engine_class(api_key, language_code)
        return engine.recognize(audio)
    except Exception as e:
        print(f"Failed to recognize speech with {engine_name} engine: {e}")
        raise e
