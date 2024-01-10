import os
import openai
from dotenv import load_dotenv

load_dotenv()

def chat_whit_gpt(command):
    openai.api_key = "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"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Complete essa história em um breve texto em apenas um pequeno parágrafo"},
            {"role": "user", "content": command},
        ]
    )
    
    return response.choices[0].message["content"]