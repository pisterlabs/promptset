import re
import os
import json
import openai
import textwrap
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=1)



openai.api_key = open_file('openaiapikey.txt')


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.3, top_p=1.0, tokens=2000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
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
            #text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    files = os.listdir('transcripts/')
    for file in files:
        if os.path.exists('clarified/%s' % file):
            print('Skipping:', file)
            continue
        transcript = open_file('transcripts/%s' % file)
        chunks = textwrap.wrap(transcript, 6000)
        output = list()
        for chunk in chunks:
            # get topics
            prompt = open_file('prompt_topic_extraction.txt').replace('<<TRANSCRIPT>>', chunk)
            topics = gpt3_completion(prompt)
            # get notes
            prompt = open_file('prompt_topic_notes.txt').replace('<<TRANSCRIPT>>', chunk).replace('<<TOPICS>>', topics)
            notes = gpt3_completion(prompt)
            #print(topics)
            #exit()
            info = {'topics': topics, 'notes': notes}
            print(info)
            output.append(info)
        filepath = 'notes/%s' % file.replace('.txt','.json')
        save_json(filepath, output)