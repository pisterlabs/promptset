"""Video Model"""
import os
from uuid import uuid4
from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
import cv2
import openai
from moviepy.editor import VideoFileClip
import models

Base = declarative_base()
openai.api_key = os.getenv("OPENAI_KEY")


class Video(Base):

    __tablename__ = "videos"
    id = Column(String(25), primary_key=True)
    filepath = Column(String(150), unique=True)
    transcript = Column(String(1024), nullable=True)
    videofile = None

    def __init__(self, name: str, filepath: str):
        """Initializes the Video model"""
        self.id = str(uuid4())
        self.name = name
        self.filepath = filepath
        models.storage.new(self)

    def save(self):
        models.storage.save()

    def connect(self):
        self.videofile = cv2.VideoCapture(self.filepath)

    def get_frame(self):
        success, frame = self.videofile.read()
        if success:
            _, jpeg = cv2.imencode(".jpg", frame)
            return jpeg.tobytes()
        return False

    @classmethod
    def transcribe(cls, filepath):
        filepath = filepath
        audio_file = "test.wav"
        # audio_file = f"{filepath.split('.')[0]}.wav"
        try:
            with VideoFileClip(filepath) as clip:
                audio = clip.audio
                audio.write_audiofile(audio_file, codec="pcm_s32le")

            with open(audio_file, "wb") as file:
                max_size = 1024 * 1024 * 24
                if os.path.getsize(audio_file) > (max_size):
                    file.truncate(max_size)
            with open(audio_file, "rb") as file:
                transcript = openai.Audio.transcribe("whisper-1", file)
                print(transcript["text"])
            if os.path.isfile(audio_file):
                os.remove(audio_file)
        except OSError as e:
            print(e)
            print("Could not write to disk")
