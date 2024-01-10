import os
import sys

import openai
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise Exception("OPENAI_API_KEY environment variable is not set")

openai.api_key = os.getenv("OPENAI_API_KEY")

languages = [
    "RUSSIAN",
    "ENGLISH",
    "SERBIAN",
]


class LanguageController:
    def get_language(self, text):
        try:
            system_prompt = (
                f"You have to answer in one word what language the text is written in, this text: {text}"
                f"""Return only in format ["language1", "language2"]: {', '.join(languages)}"""
            )
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Write me a text in your language: {text}",
                    },
                ],
            )
            print("------------------", completion, file=sys.stderr)
        except Exception as e:
            print(e)
            return "Sorry, something went wrong. Try again later"
        return completion.choices[0].message.content


languageController = LanguageController()
