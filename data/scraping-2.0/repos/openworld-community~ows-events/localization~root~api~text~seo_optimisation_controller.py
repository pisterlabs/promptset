import os

import openai
from dotenv import load_dotenv

from root.semantic_core import semantic_core

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise Exception("OPENAI_API_KEY environment variable is not set")

openai.api_key = os.getenv("OPENAI_API_KEY")


class SeoOptimisationController:
    def get_text(self, text, target_language):
        try:
            system_prompt = (
                f"You are frendly text builder which can transform text to seo optimised version."
                f"You write seo optimised texts and you use this words:"
                f" {', '.join(semantic_core[target_language])}"
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
                        "content": f"Optimise text:  {text} in {target_language} language",
                    },
                ],
            )
        except Exception as e:
            print(e)
        return completion.choices[0].message.content


seoOptimisationController = SeoOptimisationController()
