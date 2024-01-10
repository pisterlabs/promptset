import os

import openai
import whisper
import yt_dlp

openai.api_key = os.getenv("API_KEY")


class Transcriber:
    def __init__(self):
        self.prefix_path = os.path.join(os.getcwd(), "data")
        self.whisper_model = whisper.load_model("base.en")

    def get_video_id(self, video_url: str) -> str:
        return video_url.replace("https://www.youtube.com/watch?v=", "")

    def download_video(self, video_url: str) -> str:
        video_id = self.get_video_id(video_url)
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "paths": {"home": self.prefix_path},
            "outtmpl": {"default": "%(id)s.%(ext)s"},
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video_url])
            if error_code != 0:
                raise Exception("Failed to download video")

        return os.path.join(self.prefix_path, f"{video_id}.m4a")

    def summarize(self, video_url: str) -> str:
        transcript = self.transcribe(video_url)
        # Generate a summary of the transcript using OpenAI's gpt-3.5-turbo model.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this: {transcript}"},
            ],
        )
        return response["choices"][0]["message"]["content"]

    def transcribe(self, video_url: str) -> str:
        video_path = self.download_video(video_url)
        # `fp16` defaults to `True`, which tells the model to attempt to run on GPU.
        # For local demonstration purposes, we'll run this on the CPU by setting it to `False`.
        transcription = self.whisper_model.transcribe(video_path, fp16=False)
        return transcription["text"]
