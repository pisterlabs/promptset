import openai
import os
from dotenv import load_dotenv


load_dotenv()

openai.api_key = os.environ["OPEN_API_KEY"]


def generate_recommendations(prompt):
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=256,
        temperature=0.8,
    )
    return res["choices"][0]["text"].strip()
