import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def ask_gpt(prompt):

    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                            max_tokens=2048,
                                            n=1,
                                            stop=None,
                                            temperature=0.5,
                                            messages=[{
                                                'role': 'user',
                                                'content': prompt
                                            }])

    response_text = response['choices'][0]['message']['content']
    return response_text
