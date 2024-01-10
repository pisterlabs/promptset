import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPEN_API_NEW_KEY")

def get_response_from_chat_gpt_everything_bot(text):
    print("getting a response from chat gpt for the text", text)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI bot that is an expert in all areas. You can answer any question and give any advice."},
            {"role": "user", "content": text}
        ]
    )

    text = response['choices'][0]['message']['content']
    return text
