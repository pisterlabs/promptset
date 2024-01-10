import openai
from utils import read_config

openai.api_key = read_config()["openai_api_key"]

def generate_GPT(message):
    
    reseponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": message}])
    
    return reseponse["choices"][-1]["message"]["content"]
