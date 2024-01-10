import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


def gpt_completion(instruction, system_prompt=None, stop=["\n"], model_type="gpt-4"):
    if system_prompt is None:
        system_prompt = "You write paraphrases of sentences. The paraphrases are highly varied and creative."
    try:
        completion = openai.ChatCompletion.create(
            model=model_type,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},
            ],
            stop=stop,
        )
    except openai.error.OpenAIError as e:
        print(e)
        return None

    return completion.choices[0]["message"]["content"]
