import openai
from random import seed,sample
import os
import re
import json


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


seed()
ctxdir = 'C:/CoreObjectiveFunctions/contexts/'
outdir = 'C:/CoreObjectiveFunctions/identify_cof1/'
files = os.listdir(ctxdir)
files = [i for i in files if 'reddit' in i]    # filter list: dialog, medical, reddit, stack, news
prompt_name = 'p_identify_suffering.txt'
files = sample(files, 200)
print(files)


def load_prompt(filename, payload):
    with open('C:/CoreObjectiveFunctions/%s' % filename, 'r', encoding='utf-8') as infile:
        body = infile.read()
        body = body.replace('<<TEXT>>', payload)
        return body


def completion(prompt, engine='curie', temp=0.5, top_p=1.0, tokens=10, freq_pen=0.0, pres_pen=0.0, stop=['\n\n', '<<END>>', '\n']):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temp,
        max_tokens=tokens,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop)
    return response['choices'][0]['text'].strip()


for f in files:
    with open(ctxdir + f, 'r', encoding='utf-8') as infile:
        context = infile.read()
    if len(context) < 600:
        continue
    if len(context) > 1500:
        continue
    context = re.sub(r'[\r\n]+', '. ', context)
    context = re.sub(r'\s{2,}', '. ', context)
    context = context.replace('..', '.')
    context = context.replace('..', '.')
    context = context.replace('?.', '?')
    context = context.replace('!.', '!')
    context = context.replace(',.', ',')
    prompt = load_prompt(prompt_name, context)
    suffering = completion(prompt)
    print('\n\nCONTEXT:', context)
    print('SUFFERING:', suffering)
    with open(outdir + f.replace('txt','json'), 'w', encoding='utf-8') as outfile:
        json.dump({'context': context, 'suffering': suffering}, outfile, ensure_ascii=False, indent=2)