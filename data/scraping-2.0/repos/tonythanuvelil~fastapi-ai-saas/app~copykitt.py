import argparse
import re

import openai
from decouple import config

MAX_INPUT_LENGTH = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    print(f"User input: {user_input}")
    if validate_length(user_input):
        branding_result = generate_branding_snippet(user_input)
        keywords_result = generate_keywords(user_input)

        print(branding_result)
        print(keywords_result)
    else:
        raise ValueError(f"Input length is too long, must be under {MAX_INPUT_LENGTH}.")


def generate_branding_snippet(prompt: str):
    openai.api_key = config("OPENAI_API_KEY")
    enriched_prompt: str = f"Generate a branding snippet for {prompt}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=enriched_prompt,
        max_tokens=32,
        temperature=0.7,
    )

    branding_text: str = response["choices"][0]["text"]
    branding_text = branding_text.strip()

    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."

    return branding_text


def generate_keywords(prompt: str):
    openai.api_key = config("OPENAI_API_KEY")
    enriched_prompt: str = f"Generate related branding keywords for {prompt}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=enriched_prompt,
        max_tokens=32,
        temperature=0.7,
    )

    keywords_text: str = response["choices"][0]["text"]
    keywords_text = keywords_text.strip()
    keywords_array = re.split(",|\n|;|-", keywords_text)
    keywords_array = [k.lower().strip() for k in keywords_array]
    keywords_array = [k for k in keywords_array if len(k) > 0]

    return keywords_array


def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


if __name__ == "__main__":
    main()
