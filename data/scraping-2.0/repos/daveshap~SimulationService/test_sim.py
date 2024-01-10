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
ws_dir = 'worldstates/'


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


if __name__ == '__main__':
    while True:
        # load last ws
        file = os.listdir(ws_dir)[-1]
        last_ws = open_file(ws_dir + file)
        # generate event
        prompt = open_file('prompt_event.txt').replace('<<STATE>>', last_ws)
        event = gpt3_completion(prompt)
        print('\n\nEVENT:', event)
        # simulate impact
        prompt = open_file('prompt_change.txt').replace('<<STATE>>', last_ws).replace('<<EVENT>>', event)
        state_change = gpt3_completion(prompt)
        print('\n\nCHANGE:', state_change)
        # new world state
        prompt = open_file('prompt_state.txt').replace('<<STATE>>', last_ws).replace('<<EVENT>>', event).replace('<<NEW>>', state_change)
        new_ws = gpt3_completion(prompt)
        print('\n\nSTATE:', new_ws)
        # save ws
        filename = 'ws_%s.txt' % time()
        save_file(ws_dir + filename, new_ws)
        sleep(5)