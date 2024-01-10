import streamlit as st
import openai

from google.cloud import translate
from google.cloud import texttospeech
import os

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

class IdiomsAi:
    def __init__(self) -> None:
        pass

    def translate_audio(self, audio_file):
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
        transcription_text = transcription['text']
        return transcription_text

    def translate_text(self, text=None, target_language_code="es", project_id="project_id_here"):

        client = translate.TranslationServiceClient()
        location = "global"
        parent = f"projects/{project_id}/locations/{location}"

        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en-US",
                "target_language_code": target_language_code,
            }
        )

        translated_text = None
        for translation in response.translations:
            translated_text = translation.translated_text

        return translated_text

    def synthesize_text(self, text, language_code="es", voice_name="es-ES-Standard-A", voice_gender=texttospeech.SsmlVoiceGender.MALE):
        client = texttospeech.TextToSpeechClient()
        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
            ssml_gender=voice_gender,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        with open("streamlit_translated_output.mp3", "wb") as out:
            out.write(response.audio_content)

def main():
    st.title("OneTongue.ai")
    st.write("Upload an audio file to transcribe, translate, and synthesize:")
    audio_file = st.file_uploader("Choose an audio file", type=["m4a", "mp3", "wav"])

    language_codes = ['en-US', 'es-ES','ko-KR']
    language_names = ['English', 'Spanish','Korean']
    selected_language_index = st.selectbox("Select the target language", options=list(enumerate(language_names)), format_func=lambda x: x[1])[0]

    gender_options = {"Male": texttospeech.SsmlVoiceGender.MALE, "Female": texttospeech.SsmlVoiceGender.FEMALE}
    selected_gender_key = st.selectbox("Select the gender", options=list(gender_options.keys()))
    selected_gender = gender_options[selected_gender_key]


    voices = {
        ('en-US', 'Male'): ['en-US-Standard-A', 'en-US-Studio-B', 'en-US-Standard-D', 'en-US-Standard-I'],
        ('es-ES', 'Male'): ['es-ES-Standard-B', 'es-ES-Wavenet-B', 'es-US-Neural2-B', 'es-US-Neural2-C'],
        ('en-US', 'Female'): ['en-US-Standard-E', 'en-US-Studio-F', 'en-US-News-G', 'en-US-News-H'],
        ('ko-KR', 'Female'): ['ko-KR-Neural2-A', '	ko-KR-Neural2-B', '	ko-KR-Standard-A', 'ko-KR-Standard-B']
    }

    selected_voice = st.selectbox("Select the voice", options=voices[(language_codes[selected_language_index], selected_gender_key)])

    execute_button = st.button("Execute")
    translated_text = None

    if execute_button:
        if audio_file is not None:
            idioms_ai = IdiomsAi()

            transcription_text = idioms_ai.translate_audio(audio_file)
            st.write(f"Transcription text: {transcription_text}")

            translated_text = idioms_ai.translate_text(text=transcription_text, target_language_code=language_codes[selected_language_index])
            st.write(f"Translated text: {translated_text}")

            idioms_ai.synthesize_text(translated_text, language_code=language_codes[selected_language_index], voice_name=selected_voice, voice_gender=selected_gender)

        else:
            st.warning("Please upload an audio file.")

    if os.path.isfile("streamlit_translated_output.mp3"):
        if st.button("Play translated audio"):
            st.audio("streamlit_translated_output.mp3")
    else:
        st.write("Translated audio not found. Please synthesize the translated text first.")


if __name__ == "__main__":
    main()
