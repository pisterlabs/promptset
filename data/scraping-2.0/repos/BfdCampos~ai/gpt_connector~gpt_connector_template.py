import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_text(prompt):
    response = openai.Completion.create(
        engine='davinci',
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5, # high temp means more random/creative.
        presence_penalty=1.5
    )
    print(response)
    text = response.choices[0].text
    return text.strip()

prompt = "What is the meaning of life?"
text = generate_text(prompt)

