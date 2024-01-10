import argparse
import io
import os
import shutil
import tempfile
import threading
import wave
from datetime import datetime
from queue import Queue
from sys import platform

import numpy as np
import openai
import pyaudio
import scipy.io.wavfile
import torch
from gtts import gTTS
from playsound import playsound
from pydub import AudioSegment


from modules.constants import SAMPLE_RATE, CHANNELS, FORMAT, IS_DEBUG, BUFFER_SIZE, INPUT_DEVICE_INDEX, CALLBACK_INTERVAL, FRAMES_PER_BUFFER

class CyclicQueue(Queue):
    """
    A queue that maintains a fixed size, discarding the oldest item when new items are added and the queue is full.

    Attributes:
        maxsize (int): The maximum number of items the queue can hold.
    """

    def __init__(self, maxsize=BUFFER_SIZE):
        """
        Initialize the cyclic queue with a specified maximum size.

        Args:
            maxsize (int): The maximum size of the queue. When this size is reached,
                           the oldest item is automatically removed to make room for new items.
        """
        super().__init__(maxsize)

    def put(self, item, block=True, timeout=None):
        """
        Add an item to the queue. If the queue is full, the oldest item is removed to make space.

        Args:
            item: The item to be added to the queue.
            block (bool): If True, the call blocks if necessary until a free
                          slot is available. Defaults to True.
            timeout (float or None): The maximum time in seconds to block for. If None,
                                     blocks indefinitely. Defaults to None.
        """
        while self.full():
            self.get_nowait()
        super().put(item, block, timeout)
    
    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue. If the queue is empty, returns None.

        Args:
            block (bool): If True, the call blocks if necessary until an item is available.
                          Defaults to True.
            timeout (float or None): The maximum time in seconds to block for. If None,
                                     blocks indefinitely. Defaults to None.

        Returns:
            The item removed from the queue. Returns None if the queue is empty and not blocking.
        """
        return None if self.empty() else super().get(block, timeout)


class REPLY_PROCESS:
    """
    Handles the process of replying in a voice assistant system, including converting
    speech to text, generating a textual response, and converting the response back to speech.

    Attributes:
        speech_queue (Queue): Queue for storing incoming speech audio data.
        buffer_queue (CyclicQueue): Queue for buffering audio data to prevent loss.
        is_debug (bool): Flag to enable or disable debug mode.
        debug_file_path (str): Path to store debug files, if debug mode is enabled.
        debug_count (int): Counter to keep track of the number of debug files.
        client (openai.OpenAI): Client object to interact with OpenAI API.
    """

    def __init__(self):
        """
        Initialize the REPLY_PROCESS class by setting up queues, OpenAI client, and debug environment.
        """
        self.speech_queue = Queue()
        self.buffer_queue = CyclicQueue(maxsize=BUFFER_SIZE)
        self.is_debug = IS_DEBUG
        self.debug_file_path = ""
        self.debug_count = 0
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI()
        self.setup_debug_environment()

    def setup_debug_environment(self):
        """
        Configures the environment for debugging by creating necessary directories.
        """
        if self.is_debug:
            dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_folder = f"./debug/{dt_str}"
            os.makedirs(debug_folder, exist_ok=True)
            self.debug_file_path = debug_folder + "/"

    def reply_main(self):
        """
        The main method that orchestrates the process of replying. It combines audio data,
        converts it to text, generates a response, and then converts this response back to speech.
        """
        if not self.speech_queue.empty():

            # combine audio data from both queues
            wav_data = self.combine_audio_queues()

            # create temporary WAV file from audio data
            temp_wav = self.create_temp_wav(wav_data)

            # convert speech to text
            transcription = self.speech_to_text(temp_wav)

            # generate response with gpt
            response_text = self.generate_response(transcription)

            # convert response text to speech
            response_audio_path = self.text_to_speech(response_text)

            # play response audio
            self.play_audio(response_audio_path)

            # handle debug logging and file saving
            self.handle_debug(transcription, response_text, temp_wav, response_audio_path)

    def combine_audio_queues(self):
        """
        Combines audio data from both buffer and speech queues into a single numpy array.

        Returns:
            numpy.ndarray: The combined audio data as a numpy array.
        """
        wav_raws = [self.buffer_queue.get().numpy() for _ in range(self.buffer_queue.qsize())]
        wav_raws.extend([self.speech_queue.get().numpy() for _ in range(self.speech_queue.qsize())])
        return np.concatenate(wav_raws)

    def create_temp_wav(self, wav_data):
        """
        Creates a temporary WAV file from provided audio data.

        Args:
            wav_data (numpy.ndarray): The audio data to be written to the WAV file.

        Returns:
            str: The file path of the created temporary WAV file.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            scipy.io.wavfile.write(temp_file.name, SAMPLE_RATE, wav_data)
            return temp_file.name

    def speech_to_text(self, input_audio):
        """
        Converts speech in an audio file to text using OpenAI's Whisper API.

        Args:
            input_audio (str): File path of the audio file to be converted to text.

        Returns:
            str: The transcribed text from the audio.
        """
        with open(input_audio, "rb") as input_audiofile:
            response = self.client.audio.transcriptions.create(model="whisper-1", file=input_audiofile, response_format="text", language="ja")
        return response

    def generate_response(self, transcription):
        """
        Generates a textual response based on the given transcription using OpenAI's GPT model.

        Args:
            transcription (str): The transcribed text for which a response is to be generated.

        Returns:
            str: The generated response text.
        """
        response = self.client.chat.completions.create(model="gpt-4", messages=[{'role': 'system', 'content': 'You are my friend. Please respond to the conversation.'}, {"role": "user", "content": transcription}], temperature=0.0)
        return response.choices[0].message.content

    def text_to_speech(self, input_text):
        """
        Converts the provided text to speech.

        Args:
            input_text (str): The text to be converted into speech.

        Returns:
            str: The file path of the created temporary MP3 file containing the speech.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts = gTTS(text=input_text, lang="ja")
            tts.save(temp_file.name)
            sound = AudioSegment.from_mp3(temp_file.name).speedup(playback_speed=1.5)
            sound.export(temp_file.name, format="mp3")
            return temp_file.name

    def play_audio(self, file_path):
        """
        Plays the audio file at the given path in a separate thread.

        Args:
            file_path (str): The file path of the audio file to be played.
        """
        def audio_thread():
            try:
                playsound(file_path)
            except Exception as e:
                print(f"Error playing audio: {e}")

        threading.Thread(target=audio_thread).start()

    def handle_debug(self, transcription, response_text, input_audio, output_audio):
        """
        Handles debug logging and file saving if debug mode is enabled.

        Args:
            transcription (str): The transcribed text from speech.
            response_text (str): The generated textual response.
            input_audio (str): File path of the input audio.
            output_audio (str): File path of the output audio.
        """
        if self.is_debug:
            print(f"transcription: {transcription}")
            print(f"response_text: {response_text}")
            self.debug_count += 1
            shutil.copyfile(input_audio, f"{self.debug_file_path}input_{self.debug_count}.wav")
            shutil.copyfile(output_audio, f"{self.debug_file_path}output_{self.debug_count}.mp3")

