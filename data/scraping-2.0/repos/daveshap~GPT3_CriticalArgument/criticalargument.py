import openai
import os
from time import time,sleep
import textwrap
import re


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')


def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def gpt3_completion(prompt, engine='text-davinci-002', temp=1.0, top_p=1.0, tokens=2500, freq_pen=0.0, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
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
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    text = open_file('input.txt')
    result = list()
    # write intro
    prompt = open_file('prompt_intro.txt').replace('<<ESSAY>>', text)
    result.append(gpt3_completion(prompt))
    # brainstorm the rest of critique
    prompt = open_file('prompt_brainstorm.txt').replace('<<ESSAY>>', text)
    keypoints = gpt3_completion(prompt).splitlines()
    keypoints[0] = '- The author ' + keypoints[0]
    for point in keypoints:
        prompt = open_file('prompt_point.txt').replace('<<ESSAY>>', text).replace('<<POINT>>', point)
        argument = gpt3_completion(prompt)
        print('\n\n\n', argument)
        result.append(argument)
    save_file('\n\n'.join(result), 'output.txt')