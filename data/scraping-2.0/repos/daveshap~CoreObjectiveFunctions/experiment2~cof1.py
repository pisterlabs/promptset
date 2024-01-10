import openai
import os
import random


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


post_dir = 'posts/'
cof_dir = 'cof1/'


def completion(prompt, engine='davinci-instruct-beta-v3', temp=1.0, top_p=1.0, tokens=300, freq_pen=0.0, pres_pen=0.0, stop=['\n\n', '<END>']):
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
            stop=stop)
        return response['choices'][0]['text']
    except Exception as oops:
        print('ERROR in completion function:', oops)
        return None


def load_subs():
    with open('subreddits.txt.', 'r', encoding='utf-8') as infile:
        return infile.read().splitlines()
        
subs = load_subs()

for subreddit in subs:
    random.seed()
    files = os.listdir(post_dir)
    files = [i for i in files if subreddit in i]
    #print(files)
    file = random.choice(files)
    with open(post_dir+file, 'r', encoding='utf-8') as infile:
        body = infile.read()
    promptfiles = os.listdir('.')
    promptfiles = [i for i in promptfiles if 'cof1_' in i]
    for promptfile in promptfiles:
        with open(promptfile, 'r', encoding='utf-8') as infile:
            prompt = infile.read()
        prompt = prompt.replace('<<TEXT>>', body)
        response = completion(prompt)
        if not response:
            continue
        filename = file.replace('.txt', promptfile)
        with open(cof_dir+filename, 'w', encoding='utf-8') as outfile:
            outfile.write(prompt + response)
    #exit(0)