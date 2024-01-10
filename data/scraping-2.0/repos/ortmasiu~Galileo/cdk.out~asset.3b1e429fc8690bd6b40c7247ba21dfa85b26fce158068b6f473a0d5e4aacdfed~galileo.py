import os
import openai
import argparse
from ast import List
import re

MAX_INPUT_LENGTH = 280


def main():
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input

    print(f"User input: {user_input}")
    if validate_length(user_input):
        text_result = generate_summarized_text(user_input)
        keywords_result = generate_keywords(user_input)
        print(text_result)
        print(keywords_result)
    else:
        raise ValueError(
            f"Input lenght is too long. Must be under {MAX_INPUT_LENGTH}.")


def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_summarized_text(prompt: str) -> str:
    # Load API key from env
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Summarize this for a twelve-grade student:\n\n{prompt}"
    print(enriched_prompt)

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=enriched_prompt,
        temperature=0.7,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract output text
    summarized_text: str = response["choices"][0]["text"]

    last_char = summarized_text[-1]

    if last_char not in {".", "!", "?"}:
        summarized_text += "..."

    print(f"snippet: {summarized_text}")
    return summarized_text


def generate_keywords(prompt: str) -> List(str):
    # Load API key from an environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate 5 category keywords for {prompt}: "
    print(enriched_prompt)

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=enriched_prompt,
        max_tokens=50,
    )

    # Extract output text
    keywords_text: str = response["choices"][0]["text"]

    last_char = keywords_text[-1]
    if last_char not in {".", "!", "?"}:
        keywords_text += "..."

    # Remove the unwanted characters
    keywords_array = re.split(",|\n|-", keywords_text)
    keywords_array = [k.lower().strip() for k in keywords_array]
    keywords_array = [k.strip("12345).") for k in keywords_array if len(k) > 0]

    # strip whitespace
    keywords_array = [k.strip() for k in keywords_array]

    print(f"keywords: {keywords_array}")
    return keywords_array


if __name__ == "__main__":
    main()
