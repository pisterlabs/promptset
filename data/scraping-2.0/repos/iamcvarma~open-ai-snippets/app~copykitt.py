import os
import openai
import argparse
import re

MAX_INPUT_LENGTH = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',type=str, required=True)
    args = parser.parse_args()
    user_input = args.input
    if validate_length(user_input):
        generate_branding_snippet(user_input)
        generate_branding_keywords(user_input)
    else:
        raise ValueError(f'Input length is too long.Must be under {MAX_INPUT_LENGTH}. Submitted input is {user_input}')

def validate_length(prompt:str)->bool:
    return len(prompt)<=MAX_INPUT_LENGTH

def generate_branding_snippet(subject:str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Write a creative ad for {subject} to run on FaceBook"

    response = openai.Completion.create(model="davinci-instruct-beta-v3",prompt=prompt,temperature=0.5,max_tokens=60,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0)


    branding_text = response['choices'][0]['text'].strip()
    if branding_text[-1].isalpha():
        branding_text+='...'
    print(f'Snippet: {branding_text}')
    return branding_text

def generate_branding_keywords(subject:str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    prompt = f'Generate related branding keywords for {subject}'

    response = openai.Completion.create(model="davinci-instruct-beta-v3", prompt=prompt,max_tokens=32)
    keyword_text = response['choices'][0]['text'].strip()
    
    keyword_array = re.split(',|\n|;|-',keyword_text)
    keyword_array_result= list(filter(lambda x:len(x),map(lambda x:x.lower().strip(),keyword_array)))
    print(f'Keywords:{keyword_array_result}')
    return keyword_array_result

if __name__=="__main__":
    main()