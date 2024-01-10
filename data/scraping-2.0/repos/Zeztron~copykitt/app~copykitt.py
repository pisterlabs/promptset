from typing import List
import os
import openai
import argparse
import re

MAX_INPUT_LENGTH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()

    user_input = args.input
    if validate_length(user_input):
        generate_branding_snippet(user_input)
        generate_keywords(user_input)
    else:
        raise ValueError(
            f"Input length is too long. Must be under {MAX_INPUT_LENGTH} characters. Submitted input is {user_input}")


def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_branding_snippet(input: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Generate upbeat branding snippet for {input}"
    print(prompt)

    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=32)

    branding_text: str = response["choices"][0]["text"]

    branding_text = branding_text.replace("\n", "")
    if "\"" in branding_text:
        branding_text = branding_text.split("\"")[1]

    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."

    print(f"Branding Snippet: {branding_text}")
    return branding_text


def generate_keywords(input: str) -> List[str]:

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Generate related branding keywords for {input} without listing them"
    print(prompt)

    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=32)

    keyword_text: str = response["choices"][0]["text"]
    keyword_text = keyword_text.strip()
    keywords_array = re.split(",|\n|;|-", keyword_text)
    keywords_array = [k.strip() for k in keywords_array]
    keywords_array = [k for k in keywords_array if len(k) > 0]

    print(f"Keywords: {keywords_array}")
    return keywords_array


if __name__ == "__main__":
    main()
