

import openai
import os 
from dotenv import load_dotenv,find_dotenv
__ = load_dotenv(find_dotenv()) #read local .env file
openai.api_key=os.environ["OPENAI_API_KEY"]

def get_completion(prompt,model="gpt-3.5-turbo"):
    messages=[{"role":"user","content":prompt}]
    response=openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message['content']



prompt_example=f"""
Generate a list of two made-up india top population city along with their population.
provide them in JSON format with the following keys:
city_id,city,population.

"""
response=get_completion(prompt=prompt_example)
print(response)