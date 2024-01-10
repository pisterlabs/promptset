import time
from threading import Thread

import openai
import pyaudio
import wave
import audioop
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
import os

from joke_logger import log_joke
from joke_rater import joke_rater
from rating_responder import rating_responder
from speech_parser import parse_audio

"""
cursor parking lot
----



-----

"""

# listen to audio
# save file to mp3
# give it to whisper
# give whisper output to chatgpt
# play respective laughtrack (if applicable)

base_path = "../output"
executor = ThreadPoolExecutor(max_workers=10)


def create_file_name():
    return "output_%s" % int(time.time())


def start_listening():
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=32000,
                           input=True,
                           input_device_index=2,
                           frames_per_buffer=1024)

    frames = []
    silence_threshold = 500
    consecutive_silence = 0
    consecutive_silence_threshold = 40
    any_audio = False

    print('Listening!')

    while True:
        data = stream.read(1024)
        frames.append(data)

        # silence check
        rms = audioop.rms(data, 2)

        if rms < silence_threshold:
            consecutive_silence += 1
        else:
            any_audio = True
            consecutive_silence = 0

        if consecutive_silence > consecutive_silence_threshold:
            break

    stream.stop_stream()
    stream.close()
    py_audio.terminate()

    if any_audio:
        file_name = create_file_name()
        wav_name = f"{base_path}/{file_name}.wav"

        with wave.open(wav_name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(py_audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(32000)
            wav_file.writeframes(b"".join(frames))

        executor.submit(execute_joke, wav_name)

    start_listening()


def execute_joke(wav_name):
    try:
        print(wav_name)
        text = parse_audio(wav_name)
        if text != '':
            rating = joke_rater(text)
            if rating is not None:
                log_joke(text, rating)
            else:
                print(text)
            rating_responder(rating)
        os.remove(wav_name)
    except Exception as e:
        print(e)
        pass


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print(create_file_name())
    try:
        start_listening()
    finally:
        executor.shutdown()


if __name__ == '__main__':
    main()
