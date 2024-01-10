import os
import time
import openai
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class AudioParser(PatternMatchingEventHandler):
    patterns = ["*.wav"]

    def on_created(self, event):
        print(f"{event.src_path} has been added")
        
        time.sleep(10)
        audio_file= open(f"{event.src_path}", "rb")

        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        print(transcript)

