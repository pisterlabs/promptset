import pyaudio
from pydub import AudioSegment
import wave
import io
import openai
import time
import threading
import queue
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class WhisperAudioStream:
    def __init__(self, chunk_duration=2):
        self._chunk_duration = chunk_duration
        self._chunks = []

    def start_transcription(self):
        # Start a new thread to parse the audio chunks
        self._thread = threading.Thread(target=self.parse_chunks)
        self._thread.start()

        self.record_audio_chunks()

        self._thread.join(70)


    def record_audio_chunks(self, total_duration=60, chunk_duration=2):
        CHUNK = 1024  # Number of frames per buffer
        FORMAT = pyaudio.paInt16  # Format for audio chunks
        CHANNELS = 2  # Stereo
        RATE = 44100  # Standard sample rate for audio

        p = pyaudio.PyAudio()

        # Start recording
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        print("* Recording...")

        frames_buffer = []
        frames_for_chunk = int(RATE / CHUNK * chunk_duration)
        total_frames = int(RATE / CHUNK * total_duration)

        for _ in range(0, total_frames):
            data = stream.read(CHUNK)
            frames_buffer.append(data)
            
            if len(frames_buffer) == frames_for_chunk:
                # Save the current buffer to a BytesIO object
                audio_io = io.BytesIO()
                wf = wave.open(audio_io, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames_buffer))
                wf.close()
                audio_io.seek(0)
                audio_io.name = 'chunk.wav'
                self._chunks.append(audio_io)
                
                # Clear the frames buffer
                frames_buffer = []

        print("* Finished recording!")

        # Stop recording
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return self._chunks
    
    def parse_chunks(self):
        while True:
            print(len(self._chunks))
            if len(self._chunks):
                chunk = self._chunks.pop(0)
                
                transcript = openai.Audio.transcribe("whisper-1", chunk)
                print(transcript["text"], end=" ")

            time.sleep(0.5)

def save_as_mp3(audio_io, output_filename):
    # Load WAV from BytesIO object
    audio = AudioSegment.from_wav(audio_io)
    # Export as MP3
    audio.export(output_filename, format="mp3")


recorder = WhisperAudioStream()
# Call the function
chunks = recorder.start_transcription()

# Save the audio chunks as MP3 files
for i, chunk in enumerate(chunks):
    save_as_mp3(chunk, f"chunk{i}.mp3")


# import pvcheetah


# handle = pvcheetah.create("Lkg0NJt4TncZaWT2/GzqNwWP8amHuMbVTasjfzW+HAyHRPIBSvGLsQ==")

# def get_next_audio_frame():
#     while True:
#         partial_transcript, is_endpoint = handle.process(get_next_audio_frame())
#         if is_endpoint:
#             final_transcript = handle.flush()

#         print(partial_transcript)


