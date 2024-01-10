import asyncio
import pyaudio
import wave
import whisper
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
WAKE_TERM = os.getenv("WAKETERM")

# Parameters for recording audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048

# Initialize the audio interface
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Load the Whisper model
model = whisper.load_model("medium")




def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

async def record_audio(filename, duration=4):
    """
    An asynchronous generator that records audio continuously and yields
    filenames of audio chunks. Each chunk is approximately 4 seconds long.
    """
    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        yield filename

        await asyncio.sleep(0)


async def transcribe_audio():
    """
    Continuously transcribes audio from the microphone. Audio is recorded
    in chunks (approximately 4 seconds each), and each chunk is transcribed
    separately.
    """
    audio_generator = record_audio("chunk.wav")

    async for filename in audio_generator:
        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        if WAKE_TERM in result.text.lower():
            await wake_word_detected()

        print("Transcription:", result.text)


async def wake_word_detected():
    """
    Function to be called when the wake word is detected. It will listen
    for 20 seconds or until a 2-second silence is detected.
    """
    print("Wake word detected. Listening for command...")
    audio_generator = record_audio("command.wav", duration=10)

    async for filename in audio_generator:
        audio_segment = AudioSegment.from_wav(filename)
        silence_threshold = -40  # dB
        if audio_segment.dBFS < silence_threshold:
            print("Silence detected. Stopping recording...")
            break

        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        print("Command transcription:", result.text)
        response = get_completion(result.text)
        print("Response:", response)
        break


def main():
    """
    The main function of the script. Starts the transcription process.
    """
    asyncio.run(transcribe_audio())


if __name__ == "__main__":
    main()
