import openai
import string
import transcribe
import json
import sys
import os

# use gen_sub to turn audio into string
def get_audio(file: str)->str:
    return transcribe.get_text(file, openai.api_key)

# Get specification array
def get_specs(specifications: str)->list[str]:
    spec_arr = specifications.split()
    return spec_arr

# generate prompt based 
def prompt_gen(prompt: str, response: str)->str: 
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    tf = os.path.join(current_directory, 'gpt_prompt.txt')
    gpt_prompt = f'prompt: {prompt}\n\nresponse: {response}\n\n'
    with open(tf) as file:
        t = file.read()
        gpt_prompt += t
    return gpt_prompt


# Sets up openai API credentials
def set_api()->None:
    openai.api_key = '' # Paste in API key here

# Define a function to interact with the ChatGPT model
def chat_with_gpt(prompt: str)->str:
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
        top_p=1.0,
        n=1,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    if len(response.choices) > 0:
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't generate a response."

def result(response: str, prompt: str):
    # Check if the Python process has write access to the file
    set_api()
    temp = get_audio(response)
    t = ''
    with open(prompt, 'r') as file:
        t = file.read()
    g_prompt = prompt_gen(t, temp)
    
    with open('out.json', 'a+') as file:
        if file.tell() == 0: # File is empty
            data = {}
        else:
            file.seek(0) 
            data = json.load(file) # Load existing data
        
        t = chat_with_gpt(g_prompt)
        data[response[:-4].split('/')[-1]] = t # Add new data

        file.seek(0)
        file.truncate() # Clear the file
        json.dump(data, file) # Write the updated data back to the file
