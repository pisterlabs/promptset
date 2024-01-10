import os
import openai
from secret import API_KEY
import torch

openai.api_key = API_KEY

def get_completion(prompt, temp = 0.7, tok = 100, top_p = 1.0):
    res = openai.Completion.create(
            engine = "davinci",
            prompt = prompt,
            temperature = temp,
            max_tokens = tok,
            top_p = top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
            )
    return res["choices"][0]["text"]

# From text file, get a new string that contains random lines
def load_random_lines(path, n_lines = 100):
    f = open(path, 'r')
    lines = f.readlines()
    res = ""
    line_inds = torch.randint(0, len(lines), (n_lines,))
    
    for ind in line_inds:
        res += lines[ind]
    return res[:-1] # No need for last \n

def get_item_txt():
    prompt = load_random_lines("destguns.txt", n_lines = 10)
    res = get_completion(prompt)
    
    # Use <BOF> and <EOF> to split
    res = res[res.find("<BOF>") + 5 : res.find("<EOF>")]
    # Making explicit assumption that no weapon names have commas
    first_comma = res.find(",")
    second_comma = first_comma + res[first_comma + 1:].find(",") + 1

    weapon_type = res[len("type: "):first_comma]
    name = res[first_comma + 1 + len(" name: "):second_comma]
    flavor = res[second_comma + 1 + len(" description: "):]
    
    return name, weapon_type, flavor



