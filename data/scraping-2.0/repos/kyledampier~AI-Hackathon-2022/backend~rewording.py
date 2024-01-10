import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Map the audiences to different prompts
prompts = {
    1: "Summarize this for a 2nd grader using simple vocabulary and words with fewer than 5 syllables.",
    2: "Summarize this for a 5th grade student.",
    3: "Summarize this for a 20 year old college student.",
}

def reword(audience, text, printOn=False):
    """
    params:
    audience = [1, 2, 3]; audience level, maps to the appropriate prompt
    text = text to reword

    returns:
    response_text
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"{prompts[audience]}: \n\n{text}",
        max_tokens=400,
        temperature=0.7
    )

    if printOn:
        print(f"Prompt: {prompts[audience]} ")
        print(response)
        print(response["choices"][0]["text"])

    reworded_text = response["choices"][0]["text"]
    reworded_text  = reworded_text.replace('\n', '')
    return reworded_text