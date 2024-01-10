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

    print(f"User input: {user_input}")
    if not validate_length(user_input):
        raise ValueError(
            f"Input must be less than {MAX_INPUT_LENGTH} characters."
        )
    
    generate_branding_snippet(user_input)
    generate_keywords(user_input)

def validate_length(prompt: str):
    return len(prompt) <= MAX_INPUT_LENGTH

def generate_branding_snippet(prompt: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate upbeat branding snippet for {prompt}"
    print(enriched_prompt)

    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3", prompt=enriched_prompt, max_tokens=32
    )
    branding_text = response["choices"][0]["text"]
    branding_text = branding_text.strip(" \n,")

    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."
    print(f"Snippet: {branding_text}")

    return branding_text

def generate_keywords(prompt: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate related branding keywordsfor {prompt}"
    print(enriched_prompt)

    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3", prompt=enriched_prompt, max_tokens=32
    )
    keywords_text = response["choices"][0]["text"]
    keywords_array = re.split(",|\n|;|-", keywords_text)
    keywords_array = [k.lower().strip() for k in keywords_array if len(k) > 0]
    print(f"Keywords: {keywords_array}")

    return keywords_array

if __name__ == "__main__":
    main()
