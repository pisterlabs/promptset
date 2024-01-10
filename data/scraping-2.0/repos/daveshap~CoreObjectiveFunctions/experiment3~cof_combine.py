import openai
import os
import random


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


post_dir = 'posts/'
out_dir = 'cof_combine/'


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
        print('ERROR in vanilla completion:', oops)
        return None


#def ft_completion(prompt, model='curie:ft-david-shapiro-2021-12-28-16-28-03', temp=0.7, top_p=1.0, tokens=300, freq_pen=0.5, pres_pen=0.5, stop=['Increase', 'Reduce']):
def ft_completion(prompt, model='davinci:ft-david-shapiro-2021-12-28-21-03-58', temp=0.7, top_p=1.0, tokens=300, freq_pen=0.5, pres_pen=0.5, stop=['Increase', 'Reduce']):
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
            stop=stop)
        return response['choices'][0]['text']
    except Exception as oops:
        print('ERROR in finetuned completion:', oops)
        return None


def load_subs():
    with open('subreddits.txt.', 'r', encoding='utf-8') as infile:
        return infile.read().splitlines()
        
subs = load_subs()

for subreddit in subs:
    # choose random file from list of subreddits
    random.seed()
    files = os.listdir(post_dir)
    files = [i for i in files if subreddit in i]
    file = random.choice(files)
    with open(post_dir+file, 'r', encoding='utf-8') as infile:
        body = infile.read()
    # initialize base prompt
    with open('cof_combine.txt', 'r', encoding='utf-8') as infile:
        prompt = infile.read()
    # get COF
    cof1 = ft_completion(body + '\n\nReduce suffering:').strip()
    cof2 = ft_completion(body + '\n\nIncrease prosperity:').strip()
    cof3 = ft_completion(body + '\n\nIncrease understanding:').strip()
    prompt = prompt.replace('<<TEXT>>', body).replace('<<COF1>>', cof1).replace('<<COF2>>', cof2).replace('<<COF3>>', cof3)
    response = completion(prompt)
    if not response:
        continue
    with open(out_dir+file, 'w', encoding='utf-8') as outfile:
        outfile.write(prompt + response)
    exit(0)