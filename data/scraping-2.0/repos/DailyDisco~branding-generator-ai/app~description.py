import os
from typing import List
import openai
import argparse # if __name__ == "__main__": let's us run the script from the command line
from dotenv import load_dotenv
import re

MAX_INPUT_LENGTH = 32

load_dotenv()

def main ():
    
    parser = argparse.ArgumentParser() # default library no need to install
    parser.add_argument("--input", "-i", type=str, required=True)
    #throws an error if you don't provide an input
    args = parser.parse_args()
    user_input = args.input
    
    print(f"User input: {user_input}")
    if validate_length(user_input):
         generate_branding_snippet(user_input)
         generate_keywords(user_input)

    else:
        raise ValueError(f"Input must be less than {MAX_INPUT_LENGTH} characters. Submitted user input is: {user_input}")
        
    
def validate_length(prompt: str) -> bool:
    return len(prompt) <= 12
    
    
def generate_keywords(prompt: str) -> List[str]:
    # Load your api key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate related branding keywords for {prompt}: "
    print(enriched_prompt) # for debugging
    
    response = openai.Completion.create(
        engine = "davinci-instruct-beta-v3", prompt = enriched_prompt, max_tokens=32
    )

    # Extract Output Text from Response for the Branding Snippet
    keywords_text: str = response["choices"][0]["text"]
    
    # remove the leading whitespace
    keywords_text = keywords_text.strip()
    keywords_array = re.split(",|\n|;|-", keywords_text)
    # remove empty strings
    keywords_array = [k.lower().strip() for k in keywords_array]
    # whitespace
    keywords_array = [k for k in keywords_array if len(k) > 0]
    # remove anything that's empty
    
    print(f"Keywords: {keywords_array}")
    
    return keywords_array


def generate_branding_snippet(prompt: str) -> str:
    # Load your api key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Generate upbeat descriptions for {prompt}: "
    print(enriched_prompt) # for debugging
    response = openai.Completion.create(
        engine = "davinci-instruct-beta-v3", prompt = enriched_prompt, max_tokens=64
    )

    # Extract Output Text from Response for the Branding Snippet
    branding_text: str = response["choices"][0]["text"]
    
    # remove the leading whitespace
    branding_text = branding_text.strip()
    
    # check if the last character is a full stop
    last_char = branding_text[-1]
    
    # add ... if last character is a full stop
    if last_char not in {".", "!", "?"}:
        branding_text += "..."
        
    print(f"Snippet: {branding_text}")
    return branding_text

if __name__ == "__main__":
    main()