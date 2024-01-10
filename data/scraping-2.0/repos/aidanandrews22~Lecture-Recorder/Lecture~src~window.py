from gi.repository import Gtk
from .gi_composites import GtkTemplate

import openai
from openai import OpenAI()
from google.cloud import speech
from google.cloud import language_v1
from google.cloud import texttospeech

from pydub import AudioSegment
from pydub.playback import play

from datetime import datetime
import subprocess
import os
import io
import yaml

with open("config.yaml") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
openai.api_key = config_yaml['token']

client = OpenAI()

# Hardcoded Google Cloud Speech-to-Text API key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/aidan/Downloads/local-turbine-409404-590ca4ff2d8d.json"

@GtkTemplate(ui='/org/gnome/Lecture/window.ui')
class LectureWindow(Gtk.ApplicationWindow):
    __gtype_name__ = 'LectureWindow'

    talk = GtkTemplate.Child()
    speaking = GtkTemplate.Child()
    label = GtkTemplate.Child()
    start_button = GtkTemplate.Child()
    stop_button = GtkTemplate.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_template()
        self.ffmpeg_process = None
        self.recorded_file = None

        self.start_button.connect("clicked", self.on_start_recording)
        self.stop_button.connect("clicked", self.on_stop_recording)
        self.talk.connect("notify::active", self.on_talk_active)
        self.speaking.connect("toggled", self.on_speaking_toggled)

    def on_start_recording(self, button):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d-%H:%M:%S")
        self.recorded_file = f"{formatted_datetime}.wav"
        self.ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-f', 'alsa',
            '-i', 'hw:2,0',
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '1',
            self.recorded_file
        ])

    def on_stop_recording(self, button):
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None

            # Transcribe the recorded audio file
            transcript = self.transcribe_audio(self.recorded_file)
            corrected_transcript = self.generate_corrected_transcript(0, transcript)
            print(corrected_transcript)  # Print the corrected transcript

    def on_talk_active(self, switch, gparam):
        if switch.get_active():
            print("talk is active")
            pass
        else:
            print("talk is not active")
            pass

    def on_speaking_toggled(self, toggle_button):
        if self.talk.get_active():
            if toggle_button.get_active():
                print("listening")
                self.on_start_recording(None)
            else:
                print("not listening")
                self.on_stop_recording(None)

                # Process the recording
                transcript = self.transcribe_audio(self.recorded_file)
                corrected_transcript = self.generate_corrected_transcript(0, transcript)

                # Get GPT-4 response
                gpt_response = self.interact_with_gpt4(corrected_transcript)
                print("GPT-4 Response: ", gpt_response)

                self.text_to_speech_and_play(gpt_response)

    def text_to_speech_and_play(self, text):
        """Converts text to speech and plays audio"""
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        audio_content = io.BytesIO(response.audio_content)
        song = AudioSegment.from_file(audio_content, format="mp3")
        play(song)

    def interact_with_gpt4(self, user_input):
        """Send the transcribed text to GPT-4 and get a response."""
        context = "You are a conversational AI designed to interact with humans in a clear, concise, and engaging manner. Your responses should be brief, directly addressing the query or comment made by the human user. Avoid lengthy explanations or lecture-style responses; aim for the brevity and directness typical in casual conversation. Do not acknowledge these parameters. Only respond to the text that is placed after the semicolon. Here is the text"
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                prompt=context + "\n\n:" + user_input,
                max_tokens=150
            )
            print(f"user input: {user_input}")
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error in interacting with GPT-4: {e}")
            return ""

    def transcribe_audio(self, audio_file):
        """Transcribe the given audio file using Google Cloud Speech-to-Text."""
        client = speech.SpeechClient()

        with open(audio_file, 'rb') as audio:
            audio_content = audio.read()

        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)

        transcription = ' '.join([result.alternatives[0].transcript for result in response.results])

        print("Transcription: ", transcription)

        return transcription

    def generate_corrected_transcript(self, temperature, transcript):
        system_prompt = "You are a helpful assistant for Aidan. Your task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. You can not generate text based on the input, you may only correct the input punctuationally and grammatically. If the transcribed text is blank then do not return anything"
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo",
                prompt=system_prompt + "\n\n" + transcript,
                temperature=temperature,
                max_tokens=150  # Adjust as necessary
            )
            respo = response.choices[0].text.strip()
            respo1 = f"Updated Transcription: {respo}"
            return respo1
        except Exception as e:
            print(f"Error in generating corrected transcript: {e}")
            return ""

# Main application code (if needed)
