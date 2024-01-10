import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('GPT_API_KEY')

def get_gpt_response(contentapi):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "You are a storyteller, I want you to tell me about this subject without words or sentences like you are talking to me directly, like sure and here's. so tell me about " + contentapi 
        
        }],
        max_tokens = 250
    )

    response_text = completion['choices'][0]['message']
    return response_text["content"]



