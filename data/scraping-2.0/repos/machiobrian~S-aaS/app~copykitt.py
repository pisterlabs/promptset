#create an entry pointfor the app
# python3 <name> -i "<subject>" -> use argparse
import os
import argparse  # for input at the terminal
import re
from typing import List
import openai

MAX_INPUT_LEN = 32


def main():
    # print("Running Copy Kitt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    arg = parser.parse_args() #call it as an object used to fetch the argument -i
    user_input = arg.input

    print(f'\nUser Input: {user_input} \n')
    # check for len of user_input
    if validate_len(user_input):
        generate_branding_snippet(user_input)
        generate_keywords(user_input)

        
    else:
        raise ValueError(f"Input length too long. Must be under {MAX_INPUT_LEN}. Submitted input is {user_input}")

def generate_keywords(prompt: str) -> List[str]:
    #load my API key !manually
    openai.api_key = os.getenv("OPENAI_API_KEY")

    #creata a variable for {subject}
    subject = 'coffee'
    #create a variable for the prompt
    enriched_prompt = f'Generate related branding keywords for {prompt}:\n '
    print(enriched_prompt)

    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=enriched_prompt,
        max_tokens= 32
    )

    # print(response)

    # to extract the text from the enriched prompt
    keywords_text: str = response["choices"][0]["text"]

    keywords_text = keywords_text.strip() #strip the white space
    keywords_array = re.split(",|\n|;|-", keywords_text) #split on the occurence of any

    # remove whitespaces from each element
    keywords_array = [k.lower().strip() for k in keywords_array]
    # remove any empty elements
    keywords_array = [k for k in keywords_array if len(k) > 0]

    print(f"Keywords:  {keywords_array}")
    #return keywords_text #so that we can use in different places
    return keywords_array


def validate_len(prompt: str) -> bool: #make it return sth
    return len(prompt) <= MAX_INPUT_LEN
    


def generate_branding_snippet(prompt: str):
    #load my API key manually
    openai.api_key = os.getenv("OPENAI_API_KEY")
    #creata a variable for {subject}
    subject = 'welcome'
    #create a variable for the prompt
    enriched_prompt = f'Generate Upbeat Branding Snippet for {prompt}: \n'
    print(enriched_prompt)

    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=enriched_prompt,
        max_tokens= 32
    )

    # print(response)

    # to extract the text from the enriched prompt
    branding_text: str = response["choices"][0]["text"]
    branding_text = branding_text.strip() #strip the white space

    #ensure the last character is a fullstop, if not add an 
    last_char = branding_text[-1] #taking the last element

    if last_char not in {".","?","!"}:
        branding_text += "..." #append tripple dots to truncate statement


    print(f"Snippet: {branding_text} \n")
    return branding_text #so that we can use in different places


if __name__ == "__main__":
    main()

    #test done