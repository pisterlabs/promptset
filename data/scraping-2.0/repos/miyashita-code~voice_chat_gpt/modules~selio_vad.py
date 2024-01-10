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



from modules.constants import SAMPLE_RATE, CHANNELS, FORMAT, IS_DEBUG

class SILERO_VAD:
    """
    A class for Voice Activity Detection (VAD) using the Silero model.

    Attributes:
        model: The loaded Silero VAD model.
        data_queue (Queue): A queue to store audio data before VAD processing.
        speech_state (int): The current state of speech processing.
    """
    def __init__(self):
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False)

        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = utils

        print("Model loaded.\n")

        # stolage for audio datas before vad processing
        self.data_queue = Queue()

        self.STATE_BEFORE_SPEECH = 0
        self.STATE_SPEECHING = 1
        self.STATE_SPEECH_DONE = 2

        self.speech_state = self.STATE_BEFORE_SPEECH


    def manage_state(self, speech_timestamps):
        """
        Manages the state of speech processing based on detected speech timestamps.

        Args:
            speech_timestamps (list): A list of speech timestamps detected by the VAD.
        """

        # some timestamps are detected, go to speech state
        if speech_timestamps:
            self.speech_state = self.STATE_SPEECHING
        
        # no timestamps are detected, especially when the speech is finished, go to speech done state
        else:
            if self.speech_state == self.STATE_BEFORE_SPEECH:
                pass

            else:
                self.speech_state = self.STATE_SPEECH_DONE
