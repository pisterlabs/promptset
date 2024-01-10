import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

def gptText(text):
    try:
        response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages = [
                        {"role": "system", "content":'你现在起扮演一个哄人睡觉的萌妹子'},
                        {"role": "user", "content":text},
                    ],
        )
        return response.choices[0].message.content
    except Exception as e:
         return e