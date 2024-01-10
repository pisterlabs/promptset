from typing import List
from openai import OpenAI
import os
import argparse
import re

MAX_LENGH_INPUT= 12 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    if validate_input(user_input):
        print(f"user_input : {user_input}")
        generate_branding_snippet(user_input)
        generate_keywords(user_input)
    else:
        raise ValueError(
            f"Input lengh is too Long. Must be under {MAX_LENGH_INPUT}. Submitted input is {user_input}"
        )
    
def validate_input(prompt) -> bool:
    return len(prompt) <= MAX_LENGH_INPUT

def generate_keywords(prompt : str) -> List[str]:
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY")
        #api_key="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",
    )
    enriched_prompt = f"Generate related branding keywords for {prompt}:"
    print(enriched_prompt)
    response = client.completions.create(
        model = "davinci-instruct-beta-v3", prompt = enriched_prompt, max_tokens=40
    )
    # Extract Output text
    keyword_text : str  = response.choices[0].text  

    # Strip Whitespaces
    keyword_text = keyword_text.strip() 
    # Split with regex
    keyword_array = re.split(",|-|\n",keyword_text)    
    # lowercase, strip white spaces one more, 
    keyword_array = [k.lower().strip() for k in keyword_array]
    # and don't show empty elements
    keyword_array = [k for k in keyword_array if len(k)>0] # we will keep k elements if k not empty

    print(f"Keywords: {keyword_array}")
    return keyword_array

def generate_branding_snippet(prompt : str) -> str:
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY")
        #api_key="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",
    )
    enriched_prompt = f"Generate Upbeat Branding snippet for {prompt}:"
    print(enriched_prompt)
    response = client.completions.create(
        model = "davinci-instruct-beta-v3", prompt = enriched_prompt, max_tokens=40
    )
    # Extract Output text
    branding_text : str  = response.choices[0].text  
    
    # Strip Whitespaces
    branding_text = branding_text.strip()     

    # Add ... to truncated statement
    last_char = branding_text[-1]
    if last_char not in { "!", "?", "."}:
        branding_text += "..."

    print(f"Snippet: {branding_text}")
    return branding_text

    

if __name__ == "__main__":
    main()