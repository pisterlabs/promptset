import os
import json
import base64
import threading
import websockets
import asyncio
import pyaudio

from pynput import keyboard
import wave

from pathlib import Path
from openai import OpenAI

import logging
logger = logging.getLogger()

HERE = Path(__file__).parent


FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


pause_event = threading.Event()


def listen_for_a_sentence():
    thread = threading.Thread(target=os.system, args=(
        f"afplay {HERE / 'bling.mp3'}",))
    thread.start()

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



from pydub import AudioSegment

# Control flags and frame storage
is_recording = False
frames = []

def record_when_held():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    audio = pyaudio.PyAudio()

    # # Control flags and frame storage
    global is_recording, frames
    is_recording = False
    frames = []

    # Recording function
    def record_audio():
        global is_recording, frames, stream
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while is_recording:
            data = stream.read(CHUNK)
            frames.append(data)
    
    # Convert and save audio
    def save_audio():
        sound = AudioSegment(
            data=b''.join(frames),
            sample_width=audio.get_sample_size(FORMAT),
            frame_rate=RATE,
            channels=CHANNELS
        )
        sound.export("output.mp3", format="mp3")
        print("MP3 file saved.")

    # Key press event handler
    def on_press(key):
        global is_recording, frames
        if key == keyboard.Key.cmd:  # Change key as needed
            if not is_recording:
                is_recording = True
                frames = []
                threading.Thread(target=record_audio).start()
                print("Recording started...")
            # else:
                # is_recording = False
                # print("Stopping recording...")

    def on_release(key):
        if is_recording and key == keyboard.Key.cmd:
            print("Recording stopped. Saving file...")
            stream.stop_stream()
            stream.close()
            audio.terminate()
            # with wave.open('output.wav', 'wb') as wf:
            #     wf.setnchannels(CHANNELS)
            #     wf.setsampwidth(audio.get_sample_size(FORMAT))
            #     wf.setframerate(RATE)
            #     wf.writeframes(b''.join(frames))
            save_audio()

            return False  # Stop listener

    print("Press cmd to start recording.")
    # Start listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()





import assemblyai as aai

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