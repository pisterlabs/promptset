import os
import time
import threading
import json
import base64
import websockets
import asyncio

import logging
logger = logging.getLogger()

from pathlib import Path
HERE = Path(__file__).parent

import pyaudio
import dotenv
from pynput import keyboard
from pydub import AudioSegment
from openai import OpenAI
import assemblyai as aai
import pygame


def check_env():
    return os.getenv("OPENAI_API_KEY", None) is not None and os.getenv("ASSEMBLYAI_API_KEY", None) is not None



# Set up for audio recording (using PyAudio)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()

print(audio.get_default_input_device_info())

currently_recording = threading.Event()

RECORD_FORMAT = pyaudio.paInt16
RECORD_CHANNELS = 1
RECORD_RATE = 44100
RECORD_CHUNK = 1024

RECORDING_FRAMES = None
RECORDING_STREAM = None


def main():
    print("Looking in:", HERE.parent / ".env")
    dotenv.load_dotenv(HERE.parent / ".env")

    if not check_env():
        print("ENV vars not set in .env")
        exit(1)

    pygame.init()
    pygame.mixer.init()
    ready_tone = pygame.mixer.Sound("./bling.mp3")

    ai_voice: pygame.mixer.Sound = None

    # is_recording = False
    # currently_recording = threading.Event()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not currently_recording.is_set():
                    # is_recording = True
                    currently_recording.set()
                    ready_tone.play()

                    # start_recording()
                    threading.Thread(target=record_audio).start()

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE and currently_recording.is_set():
                    # is_recording = False
                    currently_recording.clear()

                    save_recording()
                    ready_tone.play()
                    # text = transcribe_audio()
                    # speak(text)
            elif event.type == pygame.QUIT:
                running = False

            # record_when_held()
            # text = transcribe_audio()
            # speak(text)

    pygame.quit()
    audio.terminate()





# def start_recording():
    # threading.Thread(target=record_audio).start()


def record_audio():
    global RECORDING_FRAMES, RECORDING_STREAM
    RECORDING_FRAMES = []
    RECORDING_STREAM = audio.open(format=RECORD_FORMAT, channels=RECORD_CHANNELS, rate=RECORD_RATE, input=True, frames_per_buffer=RECORD_CHUNK)

    while True:
        data = RECORDING_STREAM.read(RECORD_CHUNK)
        RECORDING_FRAMES.append(data)

        if not currently_recording.is_set():
            save_recording()
            break



def save_recording():
    global RECORDING_FRAMES, RECORDING_STREAM

    RECORDING_STREAM.stop_stream()
    RECORDING_STREAM.close()
    audio.terminate()

    sound = AudioSegment(
        data=b''.join(RECORDING_FRAMES),
        sample_width=audio.get_sample_size(RECORD_FORMAT),
        frame_rate=RECORD_RATE,
        channels=RECORD_CHANNELS
    )
    sound.export("output.mp3", format="mp3")


    








FRAMES_PER_BUFFER = 3200
pause_event = threading.Event()
def listen_for_a_sentence():
    thread = threading.Thread(target=os.system, args=(
        f"afplay {HERE / 'bling.mp3'}",))
    thread.start()

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    p = pyaudio.PyAudio()

    # Open an audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    transcription_text = asyncio.run(send_receive(stream))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return transcription_text


async def send_receive(stream):
    # print("LISTENING>>>")

    URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"
    async with websockets.connect(
            URL,
            extra_headers=(
                ("Authorization", os.getenv("ASSEMBLYAI_API_KEY")),),
            ping_interval=5,
            ping_timeout=20
    ) as _ws:

        await asyncio.sleep(0.1)  # Small delay for connection stabilization

        # Removed redundant _ws.recv() call - assuming initial handshake is not needed
        await _ws.recv()

        transcription_complete = asyncio.Event()

        async def send():
            while not transcription_complete.is_set():
                try:
                    data = stream.read(FRAMES_PER_BUFFER,
                                       exception_on_overflow=False)
                    if data:  # Check if data is not empty
                        data = base64.b64encode(data).decode("utf-8")
                        json_data = json.dumps({"audio_data": str(data)})
                        await _ws.send(json_data)
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    print(e)
                    # Log the error but don't assert false to avoid crashing
                await asyncio.sleep(0.01)

        async def receive():
            while not transcription_complete.is_set():
                try:
                    result_str = await _ws.recv()
                    result = json.loads(result_str)['text']

                    if json.loads(result_str)['message_type'] == 'PartialTranscript':
                        print(result)
                    if json.loads(result_str)['message_type'] == 'FinalTranscript':
                        # print("Human: ", result)
                        print(result)
                        transcription_complete.set()

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    print(e)
                    # Log the error but don't assert false to avoid crashing

            return result

        send_task = asyncio.create_task(send())
        receive_task = asyncio.create_task(receive())

        await asyncio.wait([send_task, receive_task], return_when=asyncio.ALL_COMPLETED)

        return await receive_task

def speak(input: str):
    HERE = Path(__file__).parent

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # print(client.models.list())

    response = client.audio.speech.create(
        model="tts-1",
        # voice="onyx",
        voice="echo",
        input=f"{input}"
    )

    speech_file_path = HERE / "speech.mp3"
    response.stream_to_file(speech_file_path)

    # play the file with afplay
    os.system(f"afplay {speech_file_path}")

    # speaking_thread = threading.Thread( target=os.system, args=(f"afplay {speech_file_path}",))
    # speaking_thread.start()
    # return speaking_thread


def transcribe_audio():
    print("Transcribing audio...")
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    transcriber = aai.Transcriber()

    filename = "./output.mp3"
    config = aai.TranscriptionConfig(speaker_labels=False)
    transcript = transcriber.transcribe(filename, config)

    print(transcript.text)

    return transcript.text

    # for utterance in transcript.utterances:
        # print(f"Speaker {utterance.speaker}: {utterance.text}")
