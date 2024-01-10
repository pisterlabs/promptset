import os
from typing import List
import openai
import argparse
import re

MAX_INPUT_LENGTH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    print(f"User input: {user_input}")

    if validate_length(user_input):
        generate_branding_snippet(user_input)
        generate_keywords(user_input)
    else:
        raise ValueError(
            f"Input must be {MAX_INPUT_LENGTH} characters or less. Submitted input was {len(user_input)} characters.")


def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_branding_snippet(prompt: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate upbeat branding snippet for {prompt}"
    print(enriched_prompt)

    response = openai.Completion.create(
        engine="davinci-instruct-beta", prompt=enriched_prompt, max_tokens=32)

    branding_text: str = response["choices"][0]["text"]
    branding_text = branding_text.strip()
    last_char = branding_text[-1]
    # add ellipsis if last char is not a sentence terminator
    if last_char not in {".", "!", "?"}:
        branding_text += "..."

    print(f"Snippet: {branding_text}")

    return branding_text


def generate_keywords(prompt: str) -> List[str]:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate related branding keywords for {prompt}"
    print(enriched_prompt)

    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3", prompt=enriched_prompt, max_tokens=32)

    keywords_text: str = response["choices"][0]["text"]
    keywords_text = keywords_text.strip()
    # split on comma, newline, asterisk, or dash to get array of keywords
    keywords_array = re.split(",|\n|;|-", keywords_text)
    keywords_array = [k.lower().strip() for k in keywords_array]
    keywords_array = [k for k in keywords_array if k != ""]

    print(f"keywords: {keywords_array}")

    return keywords_array


if __name__ == "__main__":
    main()
