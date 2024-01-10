import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('API_KEY')

# prompt GPT using song spotify song lyric transcripts
def gpt(song):
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=f"in one word describe the main theme in the following song: {song}",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response



