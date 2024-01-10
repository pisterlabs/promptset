import os
from typing import List
import openai
import argparse # if __name__ == "__main__": let's us run the script from the command line
from dotenv import load_dotenv
import re

MAX_INPUT_LENGTH = 128

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def main ():
    parser = argparse.ArgumentParser() # default library no need to install
    parser.add_argument("--input", "-i", type=str, required=True)
    #throws an error if you don't provide an input
    args = parser.parse_args()
    user_input = args.input
    
    print(f"User input: {user_input}")
    if validate_length(user_input):
        generate_codex_snippet(user_input)
        
    else:
        raise ValueError(f"Input must be less than {MAX_INPUT_LENGTH} characters. Submitted user input is: {user_input}")
    
    # print(f"User input: {user_input}")
    #      generate_gpt_snippet(user_input)
    #     #  generate_keywords(user_input)

    # else:
    #     raise ValueError(f"Input must be less than {MAX_INPUT_LENGTH} characters. Submitted user input is: {user_input}")

def validate_length(prompt: str) -> bool:
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_codex_snippet(prompt: str) -> str:
    enriched_prompt = f"Transform this Python script into JavaScript:\n ### Python\n {prompt}: \n### JavaScript"
    print(enriched_prompt) # for debugging
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt= enriched_prompt,
        temperature=0,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"]
    )
    
    # Extract Output Text from Response for the Branding Snippet
    second_language_text: str = response["choices"][0]["text"]
    
    # # remove the leading whitespace
    # second_language_text = second_language_text.strip()
    
    # # check if the last character is a full stop
    # last_char = second_language_text[-1]
    
    # # add ... if last character is a full stop
    # if last_char not in {".", "!", "?"}:
    #     second_language_text += "..."
    
    print(f"Snippet: {second_language_text}")
    return second_language_text

if __name__ == "__main__":
    main()