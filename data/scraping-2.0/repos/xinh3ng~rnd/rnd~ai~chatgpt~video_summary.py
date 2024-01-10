"""

# Usage 
export video_filepath="$HOME/dev/data/openai/video1327392268.mp4"

python ai/chatgpt/video_summary.py --video_filepath=$video_filepath

"""
from bs4 import BeautifulSoup
import json
from moviepy import editor as E
import openai
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import os
from pytube import YouTube
import requests
from totepy.generic import create_logger

logger = create_logger(__name__, level="info")

# TODO: The 'openai.organization_id' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization_id="org-bSEDQevVUvT2CCljpD3DNAEA")'
# openai.organization_id = "org-bSEDQevVUvT2CCljpD3DNAEA"


def youtube_download(url):
    youtube = YouTube(url)
    youtube = youtube.streams.get_highest_resolution()
    logger.info(f"Starting download from {url}")
    try:
        youtube.download()
        logger.info("Successfully completed the download")
    except:
        logger.error("An error has occurred")


def download(url, video_filepath):
    response = requests.get(url, stream=True)
    with open(video_filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=256):
            f.write(chunk)


def main(
    video_filepath: str = "*.mp4",
    verbose: int = 1,
):
    if 0 == 1:
        # Download from youtube
        download_url = "https://www.youtube.com/watch?v=GPOv72Awo68&ab_channel=CrashCourse"
        video_filepath = "How it Happened - The 2008 Financial Crisis Crash Course Economics 12.mp4"
        youtube_download(download_url)

    if 0 == 1:
        download_url = "https://v-cf.caffeine.tv/v1/E3E72733E4C84D8F91DF91D37E346398/abd85fae-8e1c-400a-8385-f4df54db045c/primary/chunk-stream_0-00041.m4s"
        video_filepath = "demo.mp4"
        download(download_url, video_filepath)

    logger.info("Converting mp4 to mp3 with moviepy...")
    audio_filepath = video_filepath.replace("mp4", "mp3")  # mp4 becomes mp3
    video = E.VideoFileClip(video_filepath)
    video.audio.write_audiofile(audio_filepath)

    if 1 == 1:
        logger.info("Using whisper-1 to get the transcription from a mp3 file ...")
        transcript = client.audio.transcribe("whisper-1", open(audio_filepath, "rb"))
        text = transcript["text"]
        # logger.info(text)

    logger.info("Calling chatGPT...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.",
            },
            {"role": "user", "content": f"Can you summarize the following text in about 100 words: {text}"},
        ],
    )
    logger.info("Reply: \n%s" % response["choices"][0]["message"]["content"])
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_filepath")
    parser.add_argument("--verbose", type=int, default=1)
    args = vars(parser.parse_args())
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("ALL DONE!\n")
