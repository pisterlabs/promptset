# download URL from YouTube and convert it into text files
# mp3 files which was downloaded will be deleted finally
# DONT push this file to GitHub beacause of API key

import openai
import os
import mimetypes
import logging
import sys
from dotenv import load_dotenv
from yt_dlp import YoutubeDL
import argparse


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def youtube(URLS):
    ydl_opts = {
        "format": "mp3/bestaudio/best",
        "ignoreerrors": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(URLS)
        if error_code != 0:
            logging.error("Error: " + str(error_code))
        else:
            logging.info("Downloaded")

    for files in os.listdir("."):
        if mimetypes.guess_type(files)[0] == "audio/mpeg":
            file_name = files.split(".mp3")
            video_name.append(file_name[0])


def text(video):
    f = open(f"{video}.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", f)

    with open(f"{video}.txt", "w", encoding="UTF-8") as file:
        file.write(transcript["text"])  # type: ignore
        file.close()
    os.remove(f"{video}.mp3")


if __name__ == "__main__":
    # python write_del.py --url $(PLAYLIST_URL) --title $(PLAYLIST_TITLE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="URL of the playlist")
    parser.add_argument("--title", help="Title of the playlist")
    args = parser.parse_args()
    if args.url is None:
        input("URL: ")
    if args.title is None:
        input("Title: ")

    video_name = []
    youtube(args.url)
    for video in video_name:
        text(video)
