import asyncio
import sounddevice as sd
import numpy as np
import queue
from openai import OpenAI
import wavio

# Parameters
sample_rate = 16000  # Sample rate in Hz
silence_threshold = 0.001  # Silence threshold value
chunk_size = 1024  # Chunk size for audio processing
silence_duration = 1.9  # Silence duration in seconds
chunks_per_second = sample_rate // chunk_size

# Initialize queue to store audio chunks
audio_queue = queue.Queue()

# Callback function to capture audio
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Initialize OpenAI client
client = OpenAI()

async def run_script(script_name, argument):
    process = await asyncio.create_subprocess_exec('python', script_name, argument, stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    if stdout:
        print(f"{script_name} Output: {stdout.decode()}")
    if stderr:
        print(f"{script_name} Error: {stderr.decode()}")

async def record_and_transcribe():
    with sd.InputStream(callback=audio_callback, dtype='float32', channels=1, samplerate=sample_rate, blocksize=chunk_size):
        audio_data = []
        silent_chunks = 0
        print("Recording... Speak now")

        while True:
            data = audio_queue.get()
            avg_volume = np.mean(np.abs(data))

            if avg_volume < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= silence_duration * chunks_per_second:
                    print("Silence detected, processing audio...")
                    break
            else:
                silent_chunks = 0
            audio_data.append(data)

    if audio_data:
        audio_data = np.concatenate(audio_data, axis=0)
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        wav_filename = 'recorded_audio.wav'
        wavio.write(wav_filename, audio_data, sample_rate, sampwidth=2)

        with open(wav_filename, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

            transcribed_text = transcript.text
            print("Transcription:", transcribed_text)

            await asyncio.gather(
                run_script('gpt.py', transcribed_text),
                run_script('gpti.py', transcribed_text)
            )
    else:
        print("No audio data recorded.")

async def main():
    while True:
        try:
            await record_and_transcribe()
        except KeyboardInterrupt:
            print("\nRecording and transcription stopped")
            break

if __name__ == "__main__":
    asyncio.run(main())
