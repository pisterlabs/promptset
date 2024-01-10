import os
import openai
import sys
import yt_dlp
sys.path.append('../..')

ffmpeg_path = '/usr/bin/ffmpeg'
ffprobe_path = '/usr/bin/ffprobe'

ydl_opts = {
    'ffmpeg_location': ffmpeg_path,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # This is to load your API key from an .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# ! pip install yt_dlp
# ! pip install pydub

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    # Your code to download the YouTube video/audio
    url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir="docs/youtube/"
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),
        OpenAIWhisperParser()
    )
docs = loader.load()

print(docs[0].page_content[0:500])
