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



from modules.constants import SAMPLE_RATE, CHANNELS, FORMAT, INPUT_DEVICE_INDEX, FRAMES_PER_BUFFER


class AUDIO_STREAM:
    """
    Manages the audio stream for capturing and processing audio data in real-time.

    Attributes:
        audio (pyaudio.PyAudio): The PyAudio instance for handling audio streams.
        stream (pyaudio.Stream): The active audio stream.
    """

    def __init__(self):
        """
        Initializes the AUDIO_STREAM class by creating a PyAudio instance and setting the stream to None.
        """
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def callback(self, in_data, frame_count, time_info, status, data_queue):
        """
        Callback function for the audio stream to handle incoming audio data.

        Args:
            in_data (bytes): The buffer containing the incoming audio data.
            frame_count (int): The number of frames in the buffer.
            time_info (dict): A dictionary with timing information.
            status (int): Status flag indicating if an error occurred.
            data_queue (Queue): The queue where the audio data is put for further processing.

        Returns:
            tuple: A tuple containing None and the pyaudio.paContinue flag, indicating the stream should continue.
        """
        try:
            data_queue.put(in_data)
        except Exception as e:
            print(f"Callback error: {e}")
        finally:
            return (None, pyaudio.paContinue)

    def start_stream_store_in_queue(self, data_queue):
        """
        Starts the audio stream and stores incoming audio data in the provided queue.

        Args:
            data_queue (Queue): The queue to store incoming audio data.
        """
        if not self.stream:
            self.stream = self.audio.open(format=FORMAT,
                                          rate=SAMPLE_RATE,
                                          channels=CHANNELS,
                                          input_device_index=INPUT_DEVICE_INDEX,
                                          input=True,
                                          frames_per_buffer=FRAMES_PER_BUFFER,
                                          stream_callback=lambda in_data, frame_count, time_info, status: self.callback(in_data, frame_count, time_info, status, data_queue))

        self.stream.start_stream()

    def stop_stream(self):
        """
        Stops the audio stream if it is currently active.
        """
        if self.stream:
            self.stream.stop_stream()

    def end_stream(self):
        """
        Properly closes and terminates the audio stream.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

    def look_for_audio_input(self):
        """
        Prints information about all available audio input devices.
        """
        for i in range(self.audio.get_device_count()):
            print(self.audio.get_device_info_by_index(i))
            print()

