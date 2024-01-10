import os
import io
import openai
import logging
import librosa
import tempfile
import numpy as np
import scipy.io.wavfile as wav
from google.oauth2 import service_account
from google.cloud import speech_v1 as speech

logging.basicConfig(level=logging.INFO)

class Transcription:
    def transcribe(self, audio_data: np.ndarray, sr: int, from_file: bool = False) -> str:
        raise NotImplementedError

class GoogleSpeechTranscription(Transcription):
    def __init__(self):
        client_file = "sa_speech_test.json"
        credentials = service_account.Credentials.from_service_account_file(client_file)
        self.client = speech.SpeechClient(credentials=credentials)

    def transcribe(self, audio_data: np.ndarray, sr: int, from_file: bool = False) -> str:
        if not from_file:
            audio_bytes_io = io.BytesIO(audio_data)
            audio_array, sr = librosa.load(audio_bytes_io, sr=None)
            audio_content = np.int16(audio_array * 32767).tobytes()
        else:
            audio_content = np.int16(audio_data * 32767).tobytes()

        # Prepare the audio and config objects for the API request
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sr,
            language_code="en-US",
            model="video",
        )

        # Make the API request
        try:
            response = self.client.recognize(config=config, audio=audio)
            if response.results:
                transcription = response.results[0].alternatives[0].transcript
                return transcription
            else:
                logging.warning("No transcription results returned from Google Speech API.")
                return ""
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return ""

class WhisperSpeechTranscription(Transcription):
    def __init__(self):
        openai.api_key = ''

    def transcribe(self, audio_data: np.ndarray, sr: int, from_file: bool = False) -> str:
        """ Transcribe the provided audio data using the Whisper API. """

        # Create a temporary file to store the audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav.write(temp_file.name, sr, audio_data)

        try:
            # Transcribe the audio file
            with open(temp_file.name, "rb") as f:
                response = openai.Audio.transcribe("whisper-1", file=f)

            # Check for transcription text in the response
            if response.text:
                transcription = response.text
            else:
                logging.warning("No transcription results returned from Whisper API. ⚠️")
                transcription = ""
        except Exception as e:
            logging.error(f"Error transcribing audio: {e} 🚫")
            transcription = ""
        finally:
            # Ensure temporary file is deleted
            os.remove(temp_file.name)

        return transcription

class TranscriptionService:
    def __init__(self, method: str):
        self.strategy = None
        if method == "google":
            self.strategy = GoogleSpeechTranscription()
        elif method == "whisper":
            self.strategy = WhisperSpeechTranscription()
        else:
            raise ValueError(f"Unsupported transcription method: {method}")

    def transcribe(self, audio_data: np.ndarray, sr: int, from_file: bool = False) -> str:
        return self.strategy.transcribe(audio_data, sr, from_file)
