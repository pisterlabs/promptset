import os
import requests
import json

import openai
import requests

def call_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    message = response.choices[0]['message']['content']
    return message


def split_script(script_file):
    with open(script_file, 'r') as file:
        script = file.read()
    tide = [tidbit + "." for tidbit in script.split('. ')]
    tide[-1] = tide[-1][:-1]
    return tide

def create_audio(text):
    # Implement the logic to create audio from the text
    # Replace this placeholder with your actual code
    print(f"Creating audio for: {text}")

def call_chatgpt(text):
    # Implement the logic to call ChatGPT and get the response
    # Replace this placeholder with your actual code
    response = f"Response for: {text}"
    return response

def parse_output(text):
    parsed_list = []
    lines = text.split('\n')

    for line in lines:
        if line.strip():  # Skip empty lines
            parsed_list.append(line.split('. ', 1)[1])

    return parsed_list

def extract_image_keywords(text):
    import re
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, text)
    return matches

def download_image(keyword, query):
    params = {
        'q': query,
        'count': 1,
        'imageType': 'Transparent',
        'format': 'png'
    }
    response = requests.get('https://api.bing.microsoft.com/v7.0/images/search', headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'value' in data:
            first_image_url = data['value'][0]['contentUrl']
            image_path = keyword
            with open(image_path, 'wb') as file:
                image_response = requests.get(first_image_url)
                file.write(image_response.content)
            return image_path
    return None

def process_script(script_file):
    script_list = split_script(script_file)
    audio_list = []
    for index, item in enumerate(script_list, start=1):
        create_audio(item)
        audio_list.append(item)
    
    audio_text = '\n'.join([f'{index}. {item}' for index, item in enumerate(audio_list, start=1)])
    #chatgpt_output = call_chatgpt(audio_text)
    #parsed_output = parse_output(chatgpt_output)
    
    for item in parsed_output:
        #cue = call_chatgpt(item)
        image_keywords = extract_image_keywords(cue)
        for keyword in image_keywords:
            if not os.path.exists(keyword):
                query = keyword.replace('_', ' ').replace('.png', '')
                download_image(keyword, query)

# Usage
process_script('script.txt')    
