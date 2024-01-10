import os
import regex as re
import openai
import argparse
from typing import List
input_length = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    print(f"User input: {user_input}")
    if validate_length(user_input):
        output = generate_branding_snippet(user_input)
        keywords = generate_keywords(user_input)
        print(f"Snippet: {output}")
        print(f"Keywords: {keywords}")
    else:
        raise ValueError(f"Input length is too long. Must be under {input_length}")


def generate_branding_snippet(subject: str) -> str:
    openai.api_key = ""

    prompt = f"Generate upbeat branding snippet for {subject}"
    print(prompt)
    response = openai.Completion.create(
    engine = "davinci-instruct-beta-v3",
    prompt = prompt,
    temperature = 0.0,
    max_tokens = 32)

    #extract output text
    x: str = response['choices'][0]['text']

    #strip whitespace from truncated statements
    x = x.strip()
    last_char = x[-1]
    if last_char not in (".", "!","?"):
        x += "..."
    
    return x


def generate_keywords(subject: str) -> List[str]:
    openai.api_key = ""

    prompt = f"Generate related branding keywords for {subject}"
    print(prompt)
    response = openai.Completion.create(
    engine = "davinci-instruct-beta-v3",
    prompt = prompt,
    temperature = 0.0,
    max_tokens = 32)

    #extract output text
    x: str = response['choices'][0]['text']
    keywords_array = [x.strip().lower() for x in re.split(",|\n,|\n\n|\*|-", x) if x != ""]
    

    return keywords_array


def validate_length(subject: str) -> bool:
    return len(subject) <= input_length

if __name__ == "__main__":
    main()