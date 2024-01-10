import os
import re
from dotenv import load_dotenv
import openai
from .checker import is_profane

load_dotenv()
from .helper import santise_string

api_key = os.getenv("OPEN_AI_KEY")
openai.api_key = api_key


def query_passage(query: str):
    q = query_chat(query)
    sentence, focus = extract_info(q)
    while is_profane(q) or sentence == None or focus == None:
        q = query_chat(query)
        sentence, focus = extract_info(q)
    return santise_string(f"Sentence: {sentence}\nFocus Words: {focus}")


def query_chat(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["message"]["content"]


def extract_info(input_string):
    sentence_match = re.search(r"Sentence: (.*)", input_string)
    focus_words_match = re.search(r"Focus Words: (.*)", input_string)

    sentence = sentence_match.group(1).strip() if sentence_match else None
    focus_words = focus_words_match.group(1).strip() if focus_words_match else None

    return sentence, focus_words
