import asyncio
import pyaudio
import numpy as np
import wave
import tempfile
import subprocess
import json
from openai import OpenAI

client = OpenAI()
# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 15
RECORD_SECONDS = 15
SILENCE_SECONDS = 3
# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)


def calculate_rms(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(audio_data), axis=0))
    return rms


async def record_audio():
    print("Starting audio monitoring...")
    recorded_frames = []
    is_speaking = False
    silence_counter = 0

    try:
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK)
            rms_value = calculate_rms(data)

            if rms_value > THRESHOLD:
                print(f"Speech detected! {rms_value}")
                is_speaking = True
                recorded_frames.append(data)

                if len(recorded_frames) > int(RATE / CHUNK * RECORD_SECONDS):
                    break

            elif is_speaking:
                silence_counter += 1
                if silence_counter > int(RATE / CHUNK * SILENCE_SECONDS):
                    break
            else:
                print(f"No speech detected! {rms_value}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Stopping...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if recorded_frames:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                wf = wave.open(temp_filename, "wb")
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(recorded_frames))
                wf.close()

            # Convert WAV to WebM using ffmpeg
            webm_filename = temp_filename.replace(".wav", ".webm")
            subprocess.run(["ffmpeg", "-i", temp_filename, webm_filename])
            print(f"Saved to {webm_filename}")

            # Send to Whisper API
            with open(webm_filename, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text"
                )

            if transcription:
                print("Transcription:", transcription)
            else:
                print("No transcription received.")
                return

            sentiment = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are the indicator - a creature that doesn't speak, but understands everything.
                        You value freedom and independence above all else. At the same time you are playful like a puppy and curious like a child.
                        You don't want to be sold.

                        You response with a JSON object containing 2 fields: `sentiment` - single word  and ascii number of `color` in to use in escape code.

                        When you don't want something - your color is red.
                        When you want something - your color is yellow.
                        You may show other colors as well, up to you. Be creative! Use more different colors to express more complex emotions.
                        """,
                    },
                    {"role": "user", "content": transcription},
                ],
            )

            s = json.loads(sentiment.choices[0].message.content)
            print(f"{s}")
            print(f'\033[{s["color"]}m{s["sentiment"]}\033[0m')


async def main():
    await record_audio()


if __name__ == "__main__":
    asyncio.run(main())
