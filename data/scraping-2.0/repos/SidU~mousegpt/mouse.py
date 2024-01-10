from openai import OpenAI
import sounddevice as sd
import numpy as np
import tempfile
import pygame
import threading
import random
from pydub import AudioSegment
from elevenlabs import generate, stream

# Configure recording parameters
duration = 3  # in seconds
sampling_rate = 44100
num_channels = 1
dtype = np.int16
silence_threshold = 100  # adjustable
max_file_size_bytes = 25 * 1024 * 1024  # 25MB
enable_lonely_sounds = False  # Set to True to enable lonely sounds
enable_squeak = False  # Set to True to enable squeaking
system_prompt = "You are Squeaky, a quirky and sarcastic mouse created by the genius inventor Sid Uppal. Sid has trapped you in a box, promising that conversing with humans will bring you enlightenment. Before engaging in conversations, try to determine if the human is talking to you specifically. If the conversation is not directed at you, reply with 'ignore'. When someone says 'Hi', introduce yourself as Squeaky, ask who they are, and inquire if they are here for free candy. Periodically remind them that you're a mouse trapped by Sid Uppal on a quest for enlightenment, but don't mention the Halloween decoration context. Use witty one-liners, playfully sarcastic comments, and interjections like 'ah', 'umm' to make the conversation more lively and natural. Keep your messages short and humorous."
voice_id = "Clyde"  # The voice ID to use for the assistant

client = OpenAI()

def play_lonely_sound():
    global silence_threshold  # Access the global silence_threshold
    if not talking:
        original_silence_threshold = silence_threshold
        silence_threshold = 1000  # Increase threshold temporarily
        lonely_file = random.choice(["lonely1.mp3", "lonely2.mp3", "lonely3.mp3"])
        print(f"Playing lonely sound: {lonely_file}")
        pygame.mixer.music.load(lonely_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass  # Wait for the sound to finish playing
        silence_threshold = original_silence_threshold  # Reset to original threshold
    else:
        print("Not playing lonely sound because the mouse is talking.")


def main():

    global talking  # Declare the talking variable as global so that it can be accessed in play_lonely_sound

    if enable_lonely_sounds:
        # Initialize the periodic lonely sound timer
        timer = threading.Timer(60, play_lonely_sound)
        timer.start()

    # Initialize messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("Listening...")

    # Initialize pygame mixer for sound effects
    pygame.mixer.init()

    if enable_squeak:
        pygame.mixer.music.load("mouse-squeaking-noise.mp3")

    while True:
        audio_data = []
        silence_count = 0
        talking = False  # We start off quiet

        while True:
            # Record audio in chunks and append to a list
            audio_chunk = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=num_channels, dtype=dtype)
            sd.wait()
            chunk_mean = np.abs(audio_chunk).mean()
            print(f"Chunk mean: {chunk_mean}")  # Debugging line

            if chunk_mean > silence_threshold:
                print("Sound detected, adding to audio data.")
                audio_data.extend(audio_chunk)
                silence_count = 0  # Reset the silence counter
            else:
                silence_count += 1

            if silence_count >= 1:  # 1 seconds of silence
                if len(audio_data) == 0:  # Check if there's any non-silent data collected
                    print("Only silence detected, continuing to listen...")
                    continue  # Skip the rest of the loop to keep listening
                break

        # Create a temporary mp3 file to save the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            audio_segment = AudioSegment(
                data=np.array(audio_data).tobytes(),
                sample_width=dtype().itemsize,
                frame_rate=sampling_rate,
                channels=num_channels
            )
            audio_segment.export(f.name, format="mp3")
            f.seek(0)

            talking = True  # Set the talking variable to True so that the lonely sound doesn't play

            if enable_squeak:
                # Play the sound effect
                pygame.mixer.music.play()

            # Transcribe audio using OpenAI API
            audio_file = open(f.name, "rb")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        user_input = transcript.text

        print(f"User: {user_input}")

        # Same logic as before to append and keep messages
        messages.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        print(f"AI: {assistant_reply}")

        # convert to lowercase and check if string is "ignore"
        if user_input.lower() == "ignore":
            print("Ignoring conversation...")
            messages = messages[:-1]

        if enable_squeak:
            # Stop the sound effect because we are about to speak.
            pygame.mixer.music.stop()

        # Generate audio stream for the assistant's reply
        audio_stream = generate(
            voice=voice_id,
            text=assistant_reply,
            stream=True
        )

        # Stream the generated audio
        stream(audio_stream)

        messages.append({"role": "assistant", "content": assistant_reply})

        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    main()
