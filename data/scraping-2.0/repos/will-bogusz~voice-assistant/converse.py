import os
import tkinter as tk
from tkinter import Canvas
import pyaudio
import wave
from datetime import datetime
from pydub import AudioSegment
from elevenlabs import generate,  stream, set_api_key
import threading
import openai
import time
import speech_recognition as sr
from io import BytesIO

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Active Listener AI")

        self.message_history = []

        self.mp3_file_path = ""
        
        self.first_recording = True
        self.is_recording = False
        self.is_audio_ready = False

        # grey button with red square to denote record
        self.record_btn_canvas = Canvas(root, width=200, height=200, bg='white')
        self.record_btn_canvas.pack(side='left')
        self.record_btn_canvas.create_rectangle(10, 10, 190, 190, fill='gray', outline='gray')
        self.record_btn_canvas.create_oval(70, 70, 130, 130, fill='red', outline='red')
        self.record_btn_canvas.bind("<Button-1>", self.toggle_listener)
        
        # # green play arrow greyed out
        self.playback_btn_canvas = Canvas(root, width=200, height=200, bg='white')
        self.playback_btn_canvas.pack(side='right')
        self.playback_btn_canvas.create_rectangle(10, 10, 190, 190, fill='lightgray', outline='lightgray')
        self.playback_btn_canvas.create_polygon(80, 50, 80, 150, 150, 100, fill='darkgray', outline='darkgray')
        
        self.playback_btn_canvas.bind("<Button-1>", self.craft_response)

    def save_temp_audio(self, prefix, parent, ending):
        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{prefix}-{timestamp_str}.{ending}"
        file_path = os.path.join(parent, file_name)

        return file_path

    def toggle_listener(self, event):
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)
            del self.stop_listening
            # indicate that listener is off
            self.record_btn_canvas.create_rectangle(10, 10, 190, 190, fill='gray', outline='gray')
            self.record_btn_canvas.create_oval(70, 70, 130, 130, fill='red', outline='red')
        else:
            self.start_listening()

    def start_listening(self):
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_name = f"Conversation-{timestamp_str}"
        os.makedirs(self.folder_name, exist_ok=True)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
        # indicate listener is now on
        self.record_btn_canvas.create_rectangle(10, 10, 190, 190, fill='green', outline='green')

    def callback(self, recognizer, audio):
        try:
            wav_data = BytesIO(audio.get_wav_data())
            wav_file_path = self.save_temp_audio("audio", self.folder_name, "wav")
            with open(wav_file_path, 'wb') as f:
                f.write(wav_data.read())

            transcription = self.send_to_whisper(wav_file_path)

            self.craft_response(transcription)
        except Exception as e:
            print("Could not request results from Whisper API; {0}".format(e))        

    def send_to_gpt(self, transcription):
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key_path = "openrouter.txt"

        self.message_history.append({"role": "user", "content": transcription})

        messages = [
            {"role": "system", "content": "You are a helpful virtual assistant. Limit your responses to be as concise as possible unless the user specifically requests otherwise."},
        ] + self.message_history

        response = openai.ChatCompletion.create(
          #model = "",
          model="openai/gpt-3.5-turbo",
          messages=messages,
          headers={
            "HTTP-Referer": "http://bogusz.co",
          },
          stream=True,
        )

        response_content = ""

        for chunk in response:
            part = chunk['choices'][0]['delta']
            if len(part) != 0:
                text = part['content']
                response_content += text
                text = text.replace('\n', '. ')
                yield text
            else:
                self.message_history.append({"role": "assistant", "content": response_content})
                return ""

    def send_to_whisper(self, audio_file_path):
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key_path = "openai.txt"

        try:
            audio_file= open(audio_file_path, "rb")
            transcript = openai.Audio.translate("whisper-1", audio_file)
            return transcript['text']
        except FileNotFoundError:
            raise Exception(f"Audio file not found: {audio_file_path}")
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")

    def text_to_speech(self, transcript):
        with open('eleven.txt', 'r') as file:
            api_key = file.read().strip()
    
        set_api_key(api_key)
        
        
        audio_stream = generate(
            text=self.send_to_gpt(transcript),
            voice="Antoni",
            model="eleven_monolingual_v1",
            stream=True
        )
        
        stream(audio_stream)

    def craft_response(self, transcription):
        print(transcription)
        self.text_to_speech(transcription)




if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
