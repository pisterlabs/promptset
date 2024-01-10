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
            #sleep(1)


if __name__ == '__main__':
    backstory = ''
    while True:
        # load last scene
        file = os.listdir(scene_dir)[-1]
        last_scene = open_file(scene_dir + file)
        # generate event
        prompt = open_file('prompt_event.txt').replace('<<SCENE>>', last_scene).replace('<<STORY>>', backstory).replace('<<RARITY>>', 'likely')
        event = gpt3_completion(prompt)
        print('\n\nEVENT:', event)
        # new scene
        prompt = open_file('prompt_scene.txt').replace('<<SCENE>>', last_scene).replace('<<EVENT>>', event).replace('<<STORY>>', backstory)
        new_scene = gpt3_completion(prompt)
        print('\n\nSCENE:', new_scene)
        # save scene
        filename = 'scene_%s.txt' % time()
        save_file(scene_dir + filename, new_scene)
        # summarize backstory up to this point
        backstory = (backstory + ' ' + event + ' ' + new_scene).strip()
        prompt = open_file('prompt_concise_summary.txt').replace('<<STORY>>', backstory)
        backstory = gpt3_completion(prompt)
        print('\n\nBACKSTORY:', backstory)
        #sleep(5)