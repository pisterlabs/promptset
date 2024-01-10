import socket
import threading
from pydub import AudioSegment
from pydub.playback import play
import simpleaudio as sa
from openai import OpenAI
from openai import OpenAI

client = OpenAI()

def text_to_speech_stream(text, audio_playback_event):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    # Play audio chunks as they are being streamed
    for chunk in response.iter_content(chunk_size=4096):
        if audio_playback_event.is_set():  # If stop event is set, stop playing
            break
        play_obj = sa.play_buffer(chunk, num_channels=2, bytes_per_sample=2, sample_rate=22050)
        play_obj.wait_done()

def handle_client(client_socket, audio_playback_event):
    message = client_socket.recv(1024).decode()
    if message:
        print("Received message:", message)
        
        # Set the event to stop current playback
        audio_playback_event.set()
        
        # Clear the event for the next playback
        audio_playback_event.clear()
        
        # Start streaming new audio
        text_to_speech_stream(message, audio_playback_event)

    client_socket.close()

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5003))
    server_socket.listen(1)
    print("Server listening on port 5003")

    # Event object to manage playback across threads
    audio_playback_event = threading.Event()

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client_socket, audio_playback_event))
        client_thread.start()

if __name__ == "__main__":
    main()
