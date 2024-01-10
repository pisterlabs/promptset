from openai import OpenAI
import argparse
import os

MAX_INPUT_LENGTH = 32

def main():
    print("Running Copy Kitt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input

    print(f"User input: {user_input}")
    if validate_length(user_input):
        result = generate_branding_snippet(user_input)
        keywords_result = generate_keywords(user_input)
        print(result)
        print(keywords_result)
    else:
        raise ValueError(f"Input length must be under {MAX_INPUT_LENGTH} characters")

def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH

def generate_branding_snippet(prompt: str) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are the world's best marketing and advertising guru."},
        {"role": "system", "content": f"Only use between 30 and 50 characters no matter what"},
        {"role": "user", "content": f"Generate upbeat branding snippet for {prompt}"},
    ]
    )
    branding_text = completion.choices[0].message.content
    return branding_text

def generate_keywords(prompt: str) -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are the world's best marketing and advertising guru."},
        {"role": "user", "content": f"Generate 10 related branding keywords in a sentence seperated by a comma for: {prompt}. Every word should be capitalized. Do not put a period at the end."},
        # {"role": "user", "content": f"Use between 30 and 50 characters"}
    ]
    )
    keywords = completion.choices[0].message.content
    keywords_array = keywords.split(", ")
    return keywords_array

if __name__ == "__main__":
    main()