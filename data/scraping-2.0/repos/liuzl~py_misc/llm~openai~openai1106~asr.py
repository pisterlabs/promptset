import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key = os.getenv("OPENAI_API_KEY"),
)

fname = "/Users/zliu/git/py_misc/tts/hello.mp3"
#fname = "../../oneapi/tts/77b2db0119a4e6a7ebdf264da4b57ae2.wav"

result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open(fname, "rb")
)
print(result)
