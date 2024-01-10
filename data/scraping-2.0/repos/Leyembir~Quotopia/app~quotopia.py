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
        result = generate_branding_snippet(user_input)
        keywords_result = generate_keywords(user_input)

    else:
        raise ValueError(f"Input length is too long. Please enter a prompt that is less than {MAX_INPUT_LENGTH} characters. Submitted input is length:  {len(user_input)}")

def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH



def generate_keywords(prompt: str) -> List[str]:

    #Load API key from .env file or set it manually

    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate related branding keywords for {prompt}: "
    print(enriched_prompt)
    response = openai.Completion.create(engine="text-davinci-003", prompt=enriched_prompt, max_tokens=32)


    #Extracting text from response
    keywords_text: str = response["choices"][0]["text"]
    
    #Strip whitespace
    keywords_text = keywords_text.strip()
    keywords_array = re.split(",|\n|-|1.|2.|3.|4.|5.|6.", keywords_text)
    keywords_array = [k.lower().strip() for k in keywords_array]
    keywords_array = [k for k in keywords_array if len(k) > 0]

    print(f"Keywords: {keywords_array}")
    return keywords_array

def generate_branding_snippet(prompt: str) -> str:

    #Load API key from .env file or set it manually

    openai.api_key = os.getenv("OPENAI_API_KEY")

    enriched_prompt = f"Generate upbeat branding snippet for {prompt}: "
    #for debugging purposes
    print(enriched_prompt)

    response = openai.Completion.create(engine="text-davinci-003", prompt=enriched_prompt, max_tokens=32)

    #Extracting text from response
    branding_text: str = response["choices"][0]["text"]
    
    #Strip whitespace
    branding_text = branding_text.strip()

    #Add ellipsis if necessary
    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."

    print(f"Branding snippet: {branding_text}")
    return branding_text

def generate_branding_snippet_language(prompt: str, language: str) -> str:

    #Load API key from .env file or set it manually

    openai.api_key = os.getenv("OPENAI_API_KEY")

    enriched_prompt = f"Generate upbeat branding snippet for {prompt} in {language}: "
    #for debugging purposes
    print(enriched_prompt)

    response = openai.Completion.create(engine="text-davinci-003", prompt=enriched_prompt, max_tokens=32)

    #Extracting text from response
    branding_text: str = response["choices"][0]["text"]
    
    #Strip whitespace
    branding_text = branding_text.strip()

    #Add ellipsis if necessary
    last_char = branding_text[-1]
    if last_char not in {".", "!", "?"}:
        branding_text += "..."

    print(f"Branding snippet: {branding_text}")
    return branding_text

def generate_keywords_lang(prompt: str, language: str) -> List[str]:

    #Load API key from .env file or set it manually

    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate related branding keywords for {prompt} in {language}: "
    print(enriched_prompt)
    response = openai.Completion.create(engine="text-davinci-003", prompt=enriched_prompt, max_tokens=32)


    #Extracting text from response
    keywords_text: str = response["choices"][0]["text"]
    
    #Strip whitespace
    keywords_text = keywords_text.strip()
    keywords_array = re.split(",|\n|-|1.|2.|3.|4.|5.|6.", keywords_text)
    keywords_array = [k.lower().strip() for k in keywords_array]
    keywords_array = [k for k in keywords_array if len(k) > 0]

    print(f"Keywords: {keywords_array}")
    return keywords_array

if __name__ == "__main__":
    main()