# Note: Ensure that you have the OpenAI Python library v0.27.0 or above installed
import openai
import socket
import os
import threading
import queue

# Configure OpenAI API Key
openai.api_key = ''

HOST = '0.0.0.0'
PORT = 65432
WAV_FILE = "received_audio.wav"

MAX_THREADS = 2  # Adjust as per your requirements
connection_queue = queue.Queue()

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI and return the transcription."""
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    text = transcript['text']
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    )

    return completion.choices[0].message['content']

def handle_connection(conn, addr):
    with conn:
        print(f"Processing connection from {addr}")
        data_buffer = []
        while True:
            data = conn.recv(1024)
            if not data:
                break
            if "EOF" in data.decode("utf-8", errors="ignore"):
                data_buffer.append(data[:-3])
                break
            data_buffer.append(data)

        file_name = f"received_audio_{addr[0]}_{addr[1]}.wav"
        conn.sendall(b"ACK\n")
        with open(file_name, 'wb') as file:
            for data_chunk in data_buffer:
                file.write(data_chunk)

        print(f"File {file_name} saved!")
        transcription = transcribe_audio(file_name)
        print(transcription)

        os.remove(file_name)
        print(f"File {file_name} deleted!")

        conn.sendall(transcription.encode('utf-8') + b'\nEND_TRANSCRIPTION\n')

    print(f"Finished processing connection from {addr}")

def worker():
    while True:
        conn, addr = connection_queue.get()
        try:
            handle_connection(conn, addr)
        finally:
            connection_queue.task_done()

def main():
    for _ in range(MAX_THREADS):
        threading.Thread(target=worker, daemon=True).start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f'Listening on {HOST}:{PORT}')

        while True:
            conn, addr = s.accept()
            print(f"Received connection from {addr}")
            connection_queue.put((conn, addr))

if __name__ == '__main__':
    main()
