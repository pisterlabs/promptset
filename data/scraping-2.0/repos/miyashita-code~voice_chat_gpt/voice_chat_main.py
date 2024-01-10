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


from modules.audio_stream import AUDIO_STREAM
from modules.replay_process import REPLY_PROCESS
from modules.selio_vad import SILERO_VAD

from modules.constants import SAMPLE_RATE, CHANNELS, FORMAT, IS_DEBUG, BUFFER_SIZE, INPUT_DEVICE_INDEX, CALLBACK_INTERVAL, FRAMES_PER_BUFFER




def main():
    # initialize classes
    vad = SILERO_VAD()
    reply_process = REPLY_PROCESS()
    audio_stream = AUDIO_STREAM()

    # for debug
    if IS_DEBUG : audio_stream.look_for_audio_input()

    # main loop
    while(1):
        try:
            print("start vad processing")

            # start vad processing
            realtime_vad(vad, reply_process, audio_stream)

            # start reply process
            print("start reply process")

            # generate response (voice)
            reply_process.reply_main()

            # reset vad state
            vad.speech_state = vad.STATE_BEFORE_SPEECH
        except KeyboardInterrupt:
            break
    
    audio_stream.end_stream()


def realtime_vad(vad, reply_process, audio_stream):
    audio_stream.start_stream_store_in_queue(vad.data_queue)
    

    while audio_stream.stream.is_active():
        if not vad.data_queue.empty():
            data = vad.data_queue.get()

            try:
                wav_data = io.BytesIO()
                with wave.open(wav_data, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio_stream.audio.get_sample_size(FORMAT))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(data)
                wav_data.seek(0)

                wav = vad.read_audio(wav_data, sampling_rate=SAMPLE_RATE)
                speech_timestamps = vad.get_speech_timestamps(wav, vad.model, sampling_rate=SAMPLE_RATE)

                vad.manage_state(speech_timestamps)


                if IS_DEBUG:
                    print(f"speech_state: {vad.speech_state}")

                if vad.speech_state == vad.STATE_BEFORE_SPEECH:
                    reply_process.buffer_queue.put(wav)

                elif vad.speech_state == vad.STATE_SPEECHING:
                    reply_process.speech_queue.put(wav)

                else:
                    reply_process.speech_queue.put(wav) # inorder not to cut before the speech finish
                    break
                    

            except Exception as e:
                print(f"Error processing audio: {e}")

    audio_stream.stop_stream()

   

if __name__ == '__main__':
    main()

