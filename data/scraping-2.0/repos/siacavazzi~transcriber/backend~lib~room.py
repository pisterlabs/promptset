import tempfile
from openai import OpenAI
import random
import string
from .key import key
import mimetypes
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from .transcript import Transcript
from io import BytesIO
import os


class Room():

    @classmethod
    def generate_code(cls, room_codes,length=4):
        room_code = ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
        while room_code in room_codes:
            room_code = ''.join(random.choice(string.ascii_uppercase) for _ in range(length))
        return room_code
    
    @classmethod
    def get_mime_type(cls, file_storage):
        print(file_storage.filename)
        return mimetypes.guess_type(file_storage.filename)[0]
    
    @classmethod
    def get_file_extension(cls, file_storage):
        filename = secure_filename(file_storage.filename)
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else None
    
    def __init__(self, target_length=10):
        self.transcript = Transcript()
        self.activeFile = None
        self.client = OpenAI()
        self.client.api_key = key
        self.current_len = 0
        self.target_length = target_length

    def get_transcript(self, data):
        self.record(data)
        ts = self.transcribe()
        return ts
    
    def trim_audio(self):
        
        audio = AudioSegment.from_file(self.activeFile.name, format="webm")
        duration_ms = len(audio)
        threshold_ms = self.target_length * 1000
        
        if duration_ms > threshold_ms:
            start_trim = duration_ms - threshold_ms
            trimmed_audio = audio[start_trim:]
            output = trimmed_audio
        else:
            output = audio
        temp_file, temp_path = tempfile.mkstemp(suffix='.' + 'webm')
        output.export(temp_path, format='webm')

        return temp_path

    def record(self, data):
        data.seek(0)
        if self.activeFile is None:
            self.activeFile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm", mode='wb')
        self.activeFile.write(data.read())
        return self.activeFile.name
    
    def transcribe(self):
        self.current_len += 1
        path = self.trim_audio() 
        file = open(path, 'rb')
        
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=file,
            language='en'
            )
        
        os.remove(path)
        print(transcript)
        partial_ts = {}
        partial_ts['flag'] = None
        print(f"length in seconds: {self.current_len}")
        if (self.current_len == self.target_length)  or (self.current_len > self.target_length):
            self.transcript.add_chunk(transcript.text)
            self.current_len = 0
            partial_ts['flag'] = 'chunk_end'
        partial_ts['transcript'] = self.transcript.get_partial_ts(transcript.text)
        print(partial_ts)
        return partial_ts
        


