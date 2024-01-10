import requests
from time import time
from uuid import uuid4
import numpy as np
import re
import os
import openai
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = open_file('openaiapikey.txt')
scene_dir = 'scenes/'
content_prefix = 'Sensory input: '
tempo = 30


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
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
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def nexus_send(payload):  # REQUIRED: content
    url = 'http://127.0.0.1:8888/add'
    payload['content'] = content_prefix + payload['content']
    payload['microservice'] = 'simulation_input'
    payload['model'] = 'text-davinci-002'
    payload['type'] = 'sensor input'
    response = requests.request(method='POST', url=url, json=payload)
    print(response.text)


def nexus_search(payload):
    url = 'http://127.0.0.1:8888/search'
    response = requests.request(method='POST', url=url, json=payload)
    return response.json()


def nexus_bound(payload):
    url = 'http://127.0.0.1:8888/bound'
    response = requests.request(method='POST', url=url, json=payload)
    return response.json()


def nexus_match():
    url = 'http://127.0.0.1:8888/match'
    response = requests.request(method='POST', url=url)
    return response.json()


def nexus_recent():
    url = 'http://127.0.0.1:8888/recent'
    response = requests.request(method='POST', url=url)
    return response.json()


if __name__ == '__main__':
    new_scene = 'Two men are sitting at a stone chess table in Central Park. They are playing chess. The sun is shining and birds are singing. It is a summer day. Children are running and playing in the distance. Horns honking and the bustle of New York can be heard in the background.'
    nexus_send({'content': new_scene})
    backstory = new_scene
    while True:
        last_scene = new_scene
        # generate event
        prompt = open_file('prompt_event.txt').replace('<<SCENE>>', last_scene).replace('<<STORY>>', backstory).replace('<<RARITY>>', 'interesting')
        event = gpt3_completion(prompt)
        filename = '%s_event.txt' % time()
        save_file(scene_dir + filename, event)
        nexus_send({'content': event})
        # TODO - incorporate actions from the nexus
        #payload = {'lower_bound': time() - tempo, 'upper_bound': time()}
        #memories = nexus_bound(payload)
        #action = find_actions(memories)
        #if action:
        #    event = event + '\nAction I will take: %s' % action
        #print('\n\nEVENT:', event)
        # new scene
        prompt = open_file('prompt_scene.txt').replace('<<SCENE>>', last_scene).replace('<<EVENT>>', event).replace('<<STORY>>', backstory)
        new_scene = gpt3_completion(prompt)
        print('\n\nSCENE:', new_scene)
        # save scene
        filename = '%s_scene.txt' % time()
        save_file(scene_dir + filename, new_scene)
        nexus_send({'content': new_scene})
        # summarize backstory up to this point
        backstory = (backstory + ' ' + event + ' ' + new_scene).strip()
        prompt = open_file('prompt_summary.txt').replace('<<STORY>>', backstory)
        backstory = gpt3_completion(prompt)
        print('\n\nBACKSTORY:', backstory)
        # wait
        sleep(tempo)