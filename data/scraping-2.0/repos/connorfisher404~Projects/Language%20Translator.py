import os
import tkinter as tk
from subprocess import Popen, PIPE
import threading
import whisper
import openai



openai.api_key = 'openai api key'


model = whisper.load_model("small")


process = None
recording = False

def start_recording():
    global process, recording
    if not recording:  # This (ffmpeg) works on mac it may need to be changed for windows. Also change the audio source.
        command = ["ffmpeg", "-y", "-f", "avfoundation", "-i", ":Galaxy Buds2 Pro", "recording.mp3"]
        process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=10**8)
        recording = True
        text_var.set("Recording started...")


def stop_recording():
    global process, recording
    if process and recording:
        try:
            process.stdin.write(b'q')
            process.stdin.flush()
        except Exception as e:
            print("Failed to send stop command to ffmpeg:", e)
        finally:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                print("Failed to terminate ffmpeg:", e)
                process.kill()
            process = None
            recording = False
            text_var.set("Processing recording wont be long...")
            thread = threading.Thread(target=transcribe_audio)
            thread.start()


def transcribe_audio():
    global model
    text_var.set("Transcribing...")
    
    try:
        result = model.transcribe("recording.mp3", fp16=False)
        transcription.set(result["text"])
    except Exception as e:
        transcription.set(f"Transcription failed... oops: {e}")
    finally:
        text_var.set("Transcription complete:)")
        
        try:
            os.remove("recording.mp3")
            text_var.set("Recording deleted.")
        except OSError as e:
            print("Error deleting the recording:", e)

   
    translate_text(transcription.get())

def translate_text(text):
    target_language = language_entry.get().strip().capitalize()
    if target_language:
        translation_var.set(f"Translating to {target_language}...")
        root.update_idletasks()  
        try:
            response = openai.completions.create(
              model="gpt-3.5-turbo-instruct",
              prompt=f"Translate the following text to {target_language}:\n\n{text}",
              max_tokens=60
            )
            translated_text.set(response.choices[0].text.strip())
        except Exception as e:
            translated_text.set(f"Translation failed: {e}")
        translation_var.set("Translation complete.")
    else:
        translated_text.set("No target language specified.")


root = tk.Tk()
root.title("Audio Recorder and Translator")


text_var = tk.StringVar()
text_var.set("Press 'Talk' to start recording.")

transcription = tk.StringVar()
transcription.set("")


translated_text = tk.StringVar()
translated_text.set("")


translation_var = tk.StringVar()
translation_var.set("")

language_label = tk.Label(root, text="Translate to (e.g., 'Italian'):")
language_label.pack(side=tk.TOP)
language_entry = tk.Entry(root)
language_entry.pack(side=tk.TOP)


talk_button = tk.Button(root, text="Talk", command=start_recording)
talk_button.pack(side=tk.LEFT)

stop_button = tk.Button(root, text="Stop", command=stop_recording)
stop_button.pack(side=tk.RIGHT)

status_label = tk.Label(root, textvariable=text_var)
status_label.pack(side=tk.TOP)

translation_status_label = tk.Label(root, textvariable=translation_var)
translation_status_label.pack(side=tk.TOP)


original_label = tk.Label(root, text="Original text:")
original_label.pack(side=tk.LEFT)
original_text = tk.Label(root, textvariable=transcription, wraplength=300)
original_text.pack(side=tk.LEFT)

translated_label = tk.Label(root, text="Translated text:")
translated_label.pack(side=tk.RIGHT)
translated_text_label = tk.Label(root, textvariable=translated_text, wraplength=300)
translated_text_label.pack(side=tk.RIGHT)


root.mainloop()
