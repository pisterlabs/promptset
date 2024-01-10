from openai import OpenAI


class AudioGenerationService:
    def generate_audio(self, text: str, voice: str, audio_format: str) -> str:
        client = OpenAI()

        speech_file_path = "audio_files/speech.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            response_format=audio_format,
            input=text
        )

        response.stream_to_file(speech_file_path)
        return speech_file_path
