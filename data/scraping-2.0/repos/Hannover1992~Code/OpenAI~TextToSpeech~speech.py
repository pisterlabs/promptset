import socket
import io
from pydub import AudioSegment
from pydub.playback import play, _play_with_simpleaudio
import threading
import simpleaudio as sa
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Global variable to manage playback
audio_playback = None

def text_to_speech(text):
# Generate speech using OpenAI's text-to-speech model
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    # Decode the audio file
    audio_data = response.content
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

    return audio

def handle_client(client_socket):
    global audio_playback

    try:
        message = client_socket.recv(1024).decode()
        if message:
            print("Received message:", message)

            # Stop existing playback if any
            if audio_playback and audio_playback.is_playing():
                audio_playback.stop()
                audio_playback.wait_done()  # Wait for playback to stop

            # Generate new audio playback
            audio = text_to_speech(message)
            audio_playback = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )
            audio_playback.wait_done()  # Wait for playback to finish

    finally:
        client_socket.close()
        # Reset the audio_playback object after the socket is closed
        audio_playback = None


# Der Rest Ihres Hauptprogramms bleibt gleich...

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5003))
    server_socket.listen(1)
    print("Server listening on port 5003")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    main()
import socket
import io
from pydub import AudioSegment
from pydub.playback import play, _play_with_simpleaudio
import threading
import simpleaudio as sa
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Global variable to manage playback
audio_playback = None

def text_to_speech(text):
# Generate speech using OpenAI's text-to-speech model
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    # Decode the audio file
    audio_data = response.content
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

    return audio

def handle_client(client_socket):
    global audio_playback

    try:
        message = client_socket.recv(1024).decode()
        if message:
            print("Received message:", message)

            # Stop existing playback if any
            if audio_playback and audio_playback.is_playing():
                audio_playback.stop()
                audio_playback.wait_done()  # Wait for playback to stop

            # Generate new audio playback
            audio = text_to_speech(message)
            audio_playback = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate
            )
            audio_playback.wait_done()  # Wait for playback to finish

    finally:
        client_socket.close()
        # Reset the audio_playback object after the socket is closed
        audio_playback = None


# Der Rest Ihres Hauptprogramms bleibt gleich...

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5003))
    server_socket.listen(1)
    print("Server listening on port 5003")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    main()
