import openai
import os
from en_variable.set_env import set_env

set_env()

openai.api_key = os.environ.get("OPENAI_API_KEY")

def language_detection(text):
    response = openai.Completion.create(
        engine="davinci",
        prompt="This is a review classifier that detects the language of a riview. Just reply in one word. No explanation. No formats.\n\nreview: \"{}\"\n\nlanguage:".format(text),
        temperature=0,
        max_tokens=3
    )
    return response.choices
choices=language_detection("I love this app")
for choice in choices:
    print(choice.text)
