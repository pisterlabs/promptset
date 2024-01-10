import openai
import os 
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_response(prompt: any):
    response = openai.ChatCompletion.create(
        model = os.getenv("FINETUNED_MODEL_ID"),
        messages = prompt.messages
    )
    return response.choices[0].message

