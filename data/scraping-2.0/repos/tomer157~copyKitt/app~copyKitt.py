import os
import openai
import argparse
import re
from typing import List

MAX_INPUT_LENGTH = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    print(f"User input: {user_input}")
    
    if validate_length(user_input):
       branding_result = generate_branding_snippet(user_input)
       keywords_result = generate_keywords(user_input)
       
    else:
        raise ValueError(f"Input lenght is too long... must be under {MAX_INPUT_LENGTH} chars. submitted input is {user_input}")
   


def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH

def generate_keywords(prompt: str) -> List[str]: 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prmopt = f"Generate related brand keywords for  {prompt}: "
    print(enriched_prmopt)
    response = openai.Completion.create(engine="davinci-instruct-beta-v3", prompt=enriched_prmopt, max_tokens=32)
    
    
    # Extract output text.
    keywords_text: str = response["choices"][0]["text"].strip()
    
    # Strip whitespace
    keywords_text = keywords_text.strip()
    keyword_array = re.split(",|\n|;|-", keywords_text)
    keyword_array = [k.lower().strip() for k in keyword_array]
    keyword_array = [k for k in keyword_array if len(k) > 0]
    print(f"keyword: {keyword_array}")

    return keyword_array


def generate_branding_snippet(prompt: str) -> str: 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prmopt = f"Generate upbeat branding snippet for {prompt}: "
    print(enriched_prmopt)
    response = openai.Completion.create(engine="davinci-instruct-beta-v3", prompt=enriched_prmopt, max_tokens=32)
    
    # Extract output text.
    branding_text: str = response["choices"][0]["text"].strip()
    
    # Strip whitespace
    branding_text = branding_text.strip()
    last_char = branding_text[-1]

    # Add ... to truncated statements    
    if last_char not in {".", "?", "!"}:
        branding_text += "..."

    print(f"Snippet: {branding_text}")
    return branding_text    

if __name__ == "__main__":
    main()