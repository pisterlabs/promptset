import openai

from google.cloud import translate
from google.cloud import texttospeech

from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

class IdiomsAi:
    def __init__(self) -> None:
        pass

    def read_audio_file(self, file_name):
        self.file = open(f"../{file_name}", "rb")

    def translate_audio(self):
        print('Translating Audio using Whisper...')
        transcription = openai.Audio.transcribe("whisper-1", self.file)
        self.transcription_text = transcription['text']
        print('Audio Transcribed...')
        print("Initial text: {}".format(self.transcription_text))
        return self.transcription_text

    def language_translation(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        audio_file = open("output_audio.mp3", "rb")
        transcript = openai.Audio.translate("whisper-1", audio_file)
        print('TRANSLATION')
        return transcript['text']


    def translate_text(self, text=None, project_id="project_id_here"):

        if text is None:
            text = self.transcription_text

        client = translate.TranslationServiceClient()
        location = "global"
        parent = f"projects/{project_id}/locations/{location}"

        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en-US",
                "target_language_code": "es",
            }
        )

        translated_text = None
        for translation in response.translations:
            print('Translating to Spanish...')
            translated_text = translation.translated_text
            print("Translated text: {}".format(translated_text))

        return translated_text

    def synthesize_text(self,text):

        client = texttospeech.TextToSpeechClient()

        input_text = texttospeech.SynthesisInput(text=text)

        language_codes = ['en-US','es-ES']
        language_names = ['en-US-Standard-C','es-ES-Neural2-A']
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_codes[0],
            name=language_names[0],
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        # The response's audio_content is binary.
        with open("audio_output.mp3", "wb") as out:
            out.write(response.audio_content)
            print('Audio content written to file "audio_output.mp3"')


        


i = IdiomsAi()
i.read_audio_file('audio_file_here.m4a')
transcription_text = i.translate_audio()
translated_text = i.translate_text(text=transcription_text)
i.synthesize_text(translated_text)

translation = i.language_translation()
