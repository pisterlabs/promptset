from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="昔々あるところにおじいさんとおばあさんとにわのわさんが居ました。",
)

response.stream_to_file("output.mp3")