import openai

class ASR:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def transcribe(self, audio_file_path):
        with open(audio_file_path, 'rb') as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            print('Transkript: ', transcript)

        # Zugriff auf die transkribierten Daten
        transcription = transcript.text if transcript.text else "Keine Transkription gefunden."
        return transcription
