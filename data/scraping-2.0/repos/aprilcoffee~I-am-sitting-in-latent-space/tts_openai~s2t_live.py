import openai
import pyaudio
import wave
import os
import time
import subprocess
import config
# Set your OpenAI API key
openai.api_key = config.openai_api_key

# Define the function to transcribe audio data from a WAV file using subprocess
def transcribe_audio(audio_file_path):
    # Define the transcription command
    command = [
        "python", "transcribe_subprocess.py",  # Replace with the actual script name
        "--audio_file", audio_file_path,
        "--api_key", openai_api_key,
    ]

    # Run the transcription command as a subprocess
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Transcription subprocess returned non-zero exit code: {e.returncode}")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define audio settings
FORMAT = pyaudio.paInt16  # Sample format
CHANNELS = 1  # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100  # Sample rate (samples per second)
CHUNK_SIZE = 1024  # Size of audio chunks
COLLECT_DURATION = 3  # Collect audio data for 2 seconds before transcription

# Create an audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

# Initialize variables
start_time = time.time()
collected_audio_data = []

try:
    while True:
        try:
            # Read audio data from the microphone
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Append the audio data to the collected_audio_data
            collected_audio_data.append(audio_data)

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Check if the collected duration reaches the specified duration
            if elapsed_time >= COLLECT_DURATION:
                #print(elapsed_time)
                #print('transcribing...')
                # Create a WAV file to save the collected audio data locally
                audio_file_path = "collected_audio.wav"
                with wave.open(audio_file_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b"".join(collected_audio_data))

                # Start the transcription as a subprocess
                transcribe_audio(audio_file_path)

                # Calculate end time
                end_time = start_time + elapsed_time  # Duration in seconds

                # Print the time-coded subtitle
                #print(f"{int(start_time)} --> {int(end_time)}")
                #print("Transcription started in subprocess.")
                #print()

                # Reset the collected_audio_data and start time
                collected_audio_data = []
                start_time = time.time()
            
        except IOError as e:
            # Handle buffer overflow here (e.g., skip or log the error)
            print("Buffer overflow, skipping one chunk.")


except KeyboardInterrupt:
    # Stop recording on keyboard interrupt
    print("Recording stopped.")

# Clean up and close the audio stream when needed
# stream.stop_stream()
# stream.close()
# p.terminate()
