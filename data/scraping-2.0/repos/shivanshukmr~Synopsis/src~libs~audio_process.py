import openai
from libs import constant
from libs import utils
from libs.utils import get_path
import requests
import os

openai.api_key = constant.OPENAI_KEY

def get_transcript(url):
    file_name = get_path(url)
    if os.path.exists(file_name + ".mp3.txt"):
        return open(file_name + ".mp3.txt").read()


    response = requests.get(url,allow_redirects=True)
    content_type = response.headers['content-type']

    if "text/plain" in content_type:
        return response.content


    path = utils.download_audio(url)
    if utils.is_valid_filetype(path):
        print("Transcribing file.....")
        audio_file = open(path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
        open(path + ".txt", "w").write(transcript["text"])
        # os.path.rem
        return transcript["text"]
    return None
