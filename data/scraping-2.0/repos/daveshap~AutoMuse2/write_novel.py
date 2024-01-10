import os
import json
import openai
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')


def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def get_next_chunk(filepath, name):
    numbers = filepath.replace('.txt','').replace(name,'')
    number = int(numbers) + 1
    if number < 10:
        filename = '%s000%s.txt' % (name, number)
    elif number < 100:
        filename = '%s00%s.txt' % (name, number)
    elif number < 1000:
        filename = '%s0%s.txt' % (name, number)
    return open_file('chunks/%s' % filename)


def finetune_completion(prompt, temp=0.7, top_p=1.0, tokens=300, freq_pen=0.0, pres_pen=0.0, stop=['Chapter']):
    max_retry = 20
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                #engine=engine,
                model='davinci:ft-david-shapiro:novel-writer-2022-05-23-13-08-08',
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
            sleep(5)


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=300, freq_pen=0.0, pres_pen=0.0, stop=['Chapter']):
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


def make_filename(number):
    if number < 10:
        return 'chunk000%s.txt' % number
    elif number < 100:
        return 'chunk00%s.txt' % number
    elif number < 1000:
        return 'chunk0%s.txt' % number
    else:
        return 'chunk%s.txt' % number


if __name__ == '__main__':
    outline = open_file('premise.txt')
    # prime the story variable
    prompt = open_file('prompt_first.txt').replace('<<PREMISE>>', outline)
    last_chunk = gpt3_completion(prompt, engine='davinci-instruct-beta')
    print(last_chunk)
    save_file(last_chunk, 'novel/chunk0001.txt')
    # prime the summary variable
    prompt = open_file('prompt_summary.txt').replace('<<CHUNK>>', last_chunk)
    summary_so_far = gpt3_completion(prompt)
    for i in range(2,21):
        prompt = open_file('prompt_full.txt').replace('<<OUTLINE>>', outline).replace('<<SUMMARY>>', summary_so_far).replace('<<CHUNK>>', last_chunk)
        next_chunk = finetune_completion(prompt)
        print(next_chunk)
        save_file(next_chunk, 'novel/' + make_filename(i))
        # update the summary
        prompt = open_file('prompt_summary.txt').replace('<<CHUNK>>', next_chunk)
        summary_so_far = summary_so_far + gpt3_completion(prompt)
        # shorten summary so far if it's too long
        if len(summary_so_far) > 1500:
            print('shortening the summary')
            prompt = open_file('prompt_gentle.txt').replace('<<CHUNK>>', summary_so_far)
            summary_so_far = gpt3_completion(prompt)
        last_chunk = next_chunk
