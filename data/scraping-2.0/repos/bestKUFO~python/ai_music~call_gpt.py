import openai
from dotenv import load_dotenv
from django.conf import settings
import os

# .env 파일 로드
load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

