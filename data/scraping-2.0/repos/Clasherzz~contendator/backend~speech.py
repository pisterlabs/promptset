import tkinter as tk
from tkinter import filedialog
from openai import OpenAI
import requests
import os

def transcribe_audio(file_path):
    api_key = "ENTER_YOUR_API_KEY"
    client = OpenAI(api_key=api_key)
    
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            print(transcript.text)
            # Send the transcript to the Flask backend
            payload = {'message': transcript.text}
            response = requests.post('http://127.0.0.1:5000/receive', json=payload)
            if response.status_code == 200:
                print("Transcript sent to Flask backend successfully")
            else:
                print("Failed to send transcript to Flask backend")
    except Exception as e:
        print("Error:", e)

def select_file():
    file_path = filedialog.askopenfilename()
    transcribe_audio(file_path)

def create_gui():
    root = tk.Tk()
    root.title("Audio Transcription")

    label = tk.Label(root, text="Select an audio file:")
    label.pack()

    button = tk.Button(root, text="Choose File", command=select_file)
    button.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
