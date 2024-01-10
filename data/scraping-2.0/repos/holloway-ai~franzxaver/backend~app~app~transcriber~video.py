import yt_dlp
import openai
import os
import ffmpeg
from app.core.config import settings


def download_video(url, target_path):
    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(target_path / "video") + ".%(ext)s",
        "path": str(target_path),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])
    if error_code != 0:
        raise Exception("Error downloading video")
    video_file = next(target_path.glob("video.*"))
    ext = video_file.suffix[1:]
    if ext not in ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]:
        ext = "mp4"
    audio_file = video_file.parent / f"audio.{ext}"
    ffmpeg.input(str(video_file)).audio.output(str(audio_file)).run()
    return video_file, audio_file


def transcribe(file_name):
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # response_format: 'json', 'text', 'vtt', 'srt', 'verbose_json'
    with open(file_name, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            "whisper-1",
            audio_file,
            response_format="verbose_json",
            api_key=settings.OPENAI_API,
        )
    return transcript
