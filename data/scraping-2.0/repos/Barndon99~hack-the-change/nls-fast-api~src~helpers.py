import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


def query(payload):
    completion = openai.Completion.create(
        model="text-davinci-002", 
        max_tokens=2048, 
        prompt="Create a table of student responses and sentiment:" + payload + "| Method | Feelings | Sentiment | Reason |",
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion 