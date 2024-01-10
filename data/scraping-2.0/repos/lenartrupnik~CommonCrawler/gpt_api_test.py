import openai
import os
import pandas as pd
import time
import preferences

openai.api_key = preferences.OPENAI_API_KEY
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages=messages,
        temperature=0,
    )
        
    return response.choices[0].message["content"]

test_prompt = "So are you working and if so give me a witty joke"
response = get_completion(test_prompt)
print(response)