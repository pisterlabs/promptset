import io
from distutils.util import strtobool
from pathlib import Path
from typing import Literal
from time import sleep, time

import pyaudio
import requests
import soundfile as sf
from decouple import config
from openai import OpenAI
from playsound import playsound
import threading


API_KEY = config("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
current_directory = Path(__file__).parent
buffer_file_location = current_directory / "buffer.opus"
Model = Literal["tts-1", "tts-1-hd"]
Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def download_and_play(input: str, model: Model = "tts-1", voice: Voice = "onyx"):
    start_time = time()
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input,
    )
    # Note that at this point the whole audio file is downloaded and stored in memory in it's entirety. The 'stream_to_file' method is a bit of a misnomer as it doesn't actually stream the audio here but just saves it to a file.
    response.stream_to_file(f"{current_directory}/audio.mp3")
    time_until_playback = time() - start_time
    print(f"Time until playback: {time_until_playback} seconds")
    playsound(f"{current_directory}/audio.mp3")


# download_and_play(
#     "This is a test. When the test gets much longer the difference will be much more obvious as to the speed in generation times taken for the test."
# )


def stream_audio(input: str, model: Model = "tts-1", voice: Voice = "onyx"):
    start_time = time()
    py_audio = pyaudio.PyAudio()

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": "opus",
        },
        stream=True,
    )

    if response.status_code == 200:
        chunk_size = 4096
        buffer = io.BytesIO()

        def collect_data():
            with open(buffer_file_location, "wb") as f:
                chunks_written = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    buffer.write(chunk)
                    f.write(chunk)
                    chunks_written += 1
                    if chunks_written % 3 == 0:
                        print(f"Buffer size: {buffer.tell()} bytes")

        def play_audio():
            with sf.SoundFile(buffer_file_location, "r") as audio_file:
                stream = py_audio.open(
                    format=pyaudio.paInt16,
                    channels=audio_file.channels,
                    rate=audio_file.samplerate,
                    output=True,
                )

                dtype = "int16"
                data = audio_file.read(chunk_size, dtype=dtype)
                while len(data) > 0:
                    stream.write(data.tobytes())
                    data = audio_file.read(chunk_size, dtype=dtype)

                stream.stop_stream()
                stream.close()

        collect_data_thread = threading.Thread(target=collect_data, daemon=True)
        collect_data_thread.start()

        while buffer.tell() < chunk_size:
            sleep(0.2)

        time_until_playback = time() - start_time
        print(f"Time until playback: {time_until_playback} seconds")

        play_audio()

    else:
        print(f"Error: {response.status_code} - {response.text}")

    py_audio.terminate()


def talking_gpt(query, streaming: bool = False):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
    )
    content = response.choices[0].message.content
    print(content)
    try:
        print("Loading audio... (Ctrl+C to stop loading/playing)")
        if streaming:
            stream_audio(content)
        else:
            download_and_play(content)
    except KeyboardInterrupt:
        print("Stopping playback...")
        exit(0)


if __name__ == "__main__":
    while True:
        stream_or_not: str = input("Stream audio? (y/n): ")
        try:
            stream_or_not: bool = strtobool(stream_or_not)
            break
        except ValueError:
            print("Please enter y or n")
    question = input("Enter your question: ")
    talking_gpt(question, streaming=stream_or_not)
