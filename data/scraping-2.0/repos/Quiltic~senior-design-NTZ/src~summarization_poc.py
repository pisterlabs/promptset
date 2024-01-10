import os
import textwrap
import openai
import re

from time import time, sleep
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

def open_file(path):
    with open(path, 'r', encoding='utf-8') as input_file:
        return input_file.read()

def save_file(content, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

openai.api_key = open_file('api_key.txt')


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['<<END>>']):
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
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            break
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def clean_summary(summary):
    #PUT EACH BULLET ON NEW LINE

    location = [m.start() for m in re.finditer(' -', summary)]

    for val, each in enumerate(location):
        if summary[each+2].isupper():

            summary = summary[:each] + '\n' + summary[each + 1:]

    return summary

def generate_summary(btn, master):

    btn.configure(state=NORMAL)
    btn.delete("1.0", END)
    btn.insert(END, "PREPARING NOTES........")
    btn.configure(state=DISABLED)
    master.update()

    alltext = open_file('input_backup.txt')

    chunks = textwrap.wrap(alltext, 4000)
    result = list()
    btn.configure(state=NORMAL)
    btn.delete("1.0", END)
    btn.configure(state=DISABLED)
    for chunk in chunks:
        prompt = open_file('prompt.txt').replace("<<BULLET NOTES>>", chunk)
        summary = gpt3_completion(prompt)
        summary = clean_summary(summary)

        btn.configure(state=NORMAL)
        btn.insert(END, summary)
        master.update()
        btn.configure(state=DISABLED)


        print(summary)


        result.append(summary)
    
    save_file('\n'.join(result), 'output.txt')


def load_text_to_notes(btn, master):

    btn.configure(state=NORMAL)
    btn.delete("1.0", END)
    btn.insert(END, "PREPARING NOTES........")
    btn.configure(state=DISABLED)
    master.update()


    f = open("output.txt", 'r')
    text = f.read()
    f.close()

    btn.configure(state=NORMAL)
    btn.delete("1.0", END)
    btn.insert(END, text)
    btn.configure(state=DISABLED)
    master.update()


def generate_summary_training_data(infile, outfile):

    print(infile)
    print(outfile)
    print("----------------------------------------------------------------------------------")
    alltext = open_file(infile)

    chunks = str(textwrap.wrap(alltext, 4000))
    result = list()

    for chunk in chunks:

        prompt = open_file('prompt.txt').replace("<<BULLET NOTES>>", chunk)
        summary = gpt3_completion(prompt)
        summary = clean_summary(summary)



        #print(summary)


        result.append(summary)
    
    save_file('\n'.join(result), outfile)