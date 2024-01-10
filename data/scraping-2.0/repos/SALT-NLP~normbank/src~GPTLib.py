import re
import os
import openai

def clean(text, drop_keywords=set(), lower=True):
    cleaned = re.sub('^[-*0-9\(\)•.]+( )?', '', text).replace('.','').strip()
    if lower:
        cleaned = cleaned.lower()
    for drop_keyword in drop_keywords:
        cleaned = re.sub(drop_keyword, '', cleaned).strip()
    return cleaned

def extract_list(text, drop_keywords=set(), lower=True):
    lst = re.split('; |\n|, and |, ', text)
    lst = list(sorted(set([clean(x, drop_keywords=drop_keywords, lower=lower) for x in lst if len(x)])))
    return lst

def get_completion(prompt_text, 
                   engine="text-davinci-002",
                   temperature=0.7,
                   max_tokens=250,
                   top_p=1,
                   frequency_penalty=0,
                   presence_penalty=0,
                  ):
    
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt_text,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
    return response['choices'][0]['text']

def get_completion_list(prompt_text, verbose=False, drop_keywords=set(), lower=True, engine="text-davinci-002"):
    completion = get_completion(prompt_text, engine=engine)
    if verbose:
        print(completion)
    return extract_list(completion, drop_keywords, lower)
    
with open('../resources/openai_api_key.txt', 'r') as infile:
    openai.api_key = infile.readline().strip()