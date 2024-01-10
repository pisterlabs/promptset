import os

import openai
from dotenv import load_dotenv

load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

basedir = os.path.dirname(__file__)
audiopath = os.path.join(basedir, "../audio/")

def callWhisper(fname:str) -> str:
    fpath = os.path.join(audiopath, fname)
    with open(fpath, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text

if __name__=="__main__":
    result = callWhisper("小森めと.mp4")
    print(result)