import sys
import os
import spacy
import json
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import openai
from openai import OpenAI
# Idea from: https://www.youtube.com/watch?v=CluNm3OfyO8&ab_channel=IamYou
# Find the quick actions at ~/Library/Services

# Load OpenAI API key from file
os.environ["OPENAI_API_KEY"] = "<your OpenAI API key>"

# Load OpenAI API key from environment variable for security reasons
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("No OpenAI API key provided.")

# Instantiate OpenAI client
client = OpenAI()

# Get the command line arguments
args = sys.argv[1:]

# Convert the arguments to a single string
user_content = " ".join(args)

def get_lang_detector(nlp, name):
       return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

def detect_language(text):
    doc = nlp(text)
    return doc._.language["language"]

def get_reponse(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk.choices[0].delta.content or "", flush=True, end="")

# Detect the language of the user input
language = detect_language(user_content)

# Translate the text
if language == "en" or language == "ca":
	get_reponse(f"Translate the text '{user_content}' to German. Return the translation only.")
else:
	get_reponse(f"Translate the text '{user_content}' to English. Return the translation only.")
