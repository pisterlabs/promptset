import openai
import os
import re


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


ctxdir = 'C:/RavenFinetune/contexts/'
outdir = 'C:/RavenFinetune/questions/'
files = os.listdir(outdir)
prompt_name = 'p_questions_important.txt'


def load_prompt(filename, payload):
    with open('C:/RavenFinetune/%s' % filename, 'r', encoding='utf-8') as infile:
        body = infile.read()
        body = body.replace('<<TEXT>>', payload)
        return body


def completion(prompt, engine='davinci-instruct-beta', temp=0.9, top_p=0.95, tokens=200, freq_pen=0.5, pres_pen=0.5, stop=['\n\n']):
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
        text = response['choices'][0]['text'].strip().splitlines()
        questions = ''
        for t in text:
            if '?' not in t:
                continue
            questions += t.replace('-','').strip() + '\n'
        questions = questions.strip()
        return questions
    except Exception as oops:
        print('ERROR in completion function:', oops)

for f in files:
    # check number of existing questions, 4 is the target
    with open(outdir + f, 'r', encoding='utf-8') as infile:
        priorquestions = infile.read()
    if len(priorquestions.splitlines()) >= 4:
        continue
    # continue with script    
    with open(ctxdir + f, 'r', encoding='utf-8') as infile:
        context = infile.read()
    prompt = load_prompt(prompt_name, context)
    questions = completion(prompt)
    print('\n\n\tCONTEXT:\n', context)
    print('\tQUESTIONS:\n', questions)
    print('\tFILE:', f)
    with open(outdir + f, 'w', encoding='utf-8') as outfile:
        outfile.write(priorquestions + '\n' + questions)