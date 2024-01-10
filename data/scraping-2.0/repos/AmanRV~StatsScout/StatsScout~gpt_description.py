from dotenv import load_dotenv
import os
import openai


def agenerate_bio(input_text):
    return "API Calls Exceeded. Temporarily Disabled."

def generate_bio(input_text):
    load_dotenv()
    openai.api_key = os.getenv("GPT_KEY")

    prompt = 'Please give me a short summary of this player. please include their best strengths and weaknesses.  Must be strictly one paragraph.   '
    prompt += input_text

    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens = 500,
        stop=None
    )

    return response.choices[0].text.strip()
        

