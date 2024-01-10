from pytube import YouTube
from pytube.exceptions import RegexMatchError
import logging
import tempfile
from pathlib import Path

LOGGER = logging.Logger(__name__)

def extract_metadata(video_id: str) -> dict:
    url = f"https://youtu.be/{video_id}"
    metadata = {}
    try:
        yt = YouTube(url)
    except RegexMatchError:
        LOGGER.error(f"RegexMatchError for '{url}'")
        raise RegexMatchError
    itag = None
    # we only want audio files
    files = yt.streams.filter(only_audio=True)
    for file in files:
        # from audio files we grab the first audio for mp4 (eg mp3)
        if file.mime_type == "audio/mp4":
            itag = file.itag
            break
    else:
        logging.warning("NO MP3 AUDIO FOUND")
    # get the correct mp3 'stream'
    stream = yt.streams.get_by_itag(itag)

    metadata["title"] = stream.title
    metadata["filesize_mb"] = stream.filesize_mb
    metadata["tags"] = yt.keywords
    metadata["itag"] = itag
    return metadata

def get_audio_stream(video_id: str, itag: int):
    url = f"https://youtu.be/{video_id}"
    LOGGER.info(f"Attemping to download audio from {url}")
    try:
        yt = YouTube(url)
    except RegexMatchError:
        logging.error(f"RegexMatchError for '{url}'")
        raise RegexMatchError
    LOGGER.info(f"Downloading audio from {url}")
    stream = yt.streams.get_by_itag(itag)
    #filename = f"{video_id}.mp3"
    return stream

import openai
import os
import dotenv
dotenv.load_dotenv('configs/api_keys.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
from tenacity import (
    retry,
    stop_after_attempt,  # for exponential backoff
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _transcribe_with_backoff(**kwargs):
    return openai.Audio.transcribe(**kwargs)

def get_transcription(video_id, path):
    """
    Get the transcription of a video.

    Args:
        video_id (str): The ID of the video.
        path (str): The path to the directory where the audio file is stored.

    Returns:
        str: The transcription of the video.
    """
    logging.info(f"Transcribing {video_id}")
    audio_file = Path(path) / f"{video_id}.mp3"
    #check if the audio file is greater than 25MB
    if audio_file.stat().st_size > 25000000:
        #split the audio file into 25MB chunks
        #transcribe each chunk
        #combine the transcriptions
        pass
    else:
        with open(audio_file, "rb") as fb:
            LOGGER.info(f"Transcribing {video_id}. {audio_file.stat().st_size} bytes.")
            transcript = _transcribe_with_backoff(
                model="whisper-1", file=fb, language="en"
            )
            LOGGER.info(f"Transcription complete for {video_id}.")
            transcript_text = transcript["text"]
        return transcript_text
