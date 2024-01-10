'''
This is a test of OpenAI's Whisper, which converts speech to text.
'''

import openai
import json

import pyaudio
import wave
import audioop

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
START_SILENCE_THRESHOLD = 2000  # High amplitude level to detect the start of sound
END_SILENCE_THRESHOLD = 500  # Low amplitude level to detect the end of sound
SILENCE_DURATION = 2.0  # Silence duration in seconds
WAVE_OUTPUT_FILENAME = "output.wav"

def record_user():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Waiting for sound...")

    frames = []
    silence_frames = 0
    silence_consecutive_frames = int(RATE / CHUNK * SILENCE_DURATION)
    recording_started = False

    # Record audio in chunks and append to frames
    while True:
        data = stream.read(CHUNK)

        # Calculate the amplitude (RMS)
        rms = audioop.rms(data, 2)

        # Before recording has started, use the start silence threshold
        if not recording_started:
            if rms >= START_SILENCE_THRESHOLD:
                recording_started = True
                silence_frames = 0
                print("Recording started...")

        # Once recording has started, use the end silence threshold
        else:
            if rms < END_SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0
            
            # If we've had enough consecutive silence, break the loop
            if silence_frames > silence_consecutive_frames:
                break

            # Append the frames
            frames.append(data)

    # Stop and close the stream
    print("Done recording.")
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio():
    # Note: you need to be using OpenAI Python v0.27.0 for the code below to work
    audio_file= open(WAVE_OUTPUT_FILENAME, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript.get('text')


def main():
    # read credentials from file
    with open("credentials.json", "r") as f:
        credentials = json.load(f)

    # set openai api key
    openai.api_key = credentials['openai_key']

    record_user()

    text = transcribe_audio()

    print(text)    

if __name__ == '__main__':
    main()