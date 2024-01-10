import openai
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variable
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY in the .env file.")
openai.api_key = API_KEY


SYSTEM_MESSAGE = """
"you are a sophisticated parsing entity, able to capture the distinct nuances of my specific writing style. then you are tasked to extract atomic concepts from the text using <br> to delimit each individual concept. the sum of all concepts should approximately have the same length and feel of the original input. also try to also add a hint of the context of the input text to the extracted individual concepts. remember that the text might just be a text dump from some website. try your best"
"""

with open("concept_split.ppt", "r") as file:
    PREPROMPT = file.read()

def split_concepts(text):
    # Regular expression to match content between <concept></concept> tags
    pattern = r'<concept>(.*?)</concept>'
    concepts = re.findall(pattern, text, re.DOTALL)
    return concepts

def get_concepts(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
         ]
    )
    
    return response.choices[0]["message"]["content"]

def extract_atomic_concepts(text, filename):
    """
    Function to extract atomic concepts from a custom prompt filetype using the OpenAI API.
    
    Returns:
    - str: The extracted and elaborated concepts.
    """
    
    prompt = PREPROMPT.replace("<text></text>", f"<text>{text}</text>")

    notes = split_concepts(get_concepts(prompt))
    print(notes)

    notes.append(text) 

    print(prompt)

    print(notes)

    return notes