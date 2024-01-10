import os
import requests
import openai

# Function to split the file into 25MB sections
def split_file(file_path):
    chunk_size = 25 * 1024 * 1024  # 25MB
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as file:
        part_num = 1
        while True:
            data = file.read(chunk_size)
            if not data:
                break

            # Send the chunk to the API
            send_chunk(data, file_name, file_size, part_num)

            part_num += 1

def send_chunk(data, file_name, filesize, part_num):
    audio_file= open("audio/in/output.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

# Main function to process the file
def process_file(file_path):
    if not os.path.isfile(file_path):
        print("File not found")
        return

    split_file(file_path)

if __name__ == '__main__':
    file_path = 'docs/in/output.wav'  # Replace with the path to your .wav file
    process_file(file_path)
