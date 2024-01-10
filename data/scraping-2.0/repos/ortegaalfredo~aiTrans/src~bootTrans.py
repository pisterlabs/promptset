#!/usr/bin/python
# -*- coding: utf-8 -*-
# This is the transpiler 'bootloader', it was written by hand.
# It is used to use the AI to translate the pseudocode of itself to the final transpiler.
# After this, the transpiler can transpile itself.

import os,argparse,sys
import openai


def check_api_key_validity(key):
    try:
        openai.api_key = key
        print("OpenAI API key is valid", file=sys.stderr)
    except openai.OpenAIError:
        print("Invalid OpenAI API key", file=sys.stderr)
        exit()

# ---------- OpenAI interface
# The bootloader only supports using chatGPT as transpiler. 

def call_AI_chatGPT(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a excellent programmer. Write the code to execute the given task. Always write only the raw code and nothing more, no quotes. Never write english, nor code delimiters.",
        },
        {"role": "user", "content": prompt},
    ]
    model = "gpt-3.5-turbo"
    temperature = 0
    max_tokens = 2048
    response = openai.ChatCompletion.create(
        messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
    )
    response = response.choices[0]
    response = response["message"]
    response = response["content"]
    return response
    

# Increse to 1.0 to introduce randomness in answers
temperature = 0
# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", dest="source")
parser.add_argument("-l", dest="language",default="python")
parser.add_argument("-a", action="store_true", dest="allfile")
args = parser.parse_args()

# import api key
api_key = os.environ.get("OPENAI_API_KEY")
if (api_key is None) or (len(api_key)==0): # try to load apikey from file
    try:
        api_key=open('api-key.txt','rb').read().strip().decode()
    except:
        print("Couldn't load OpenAI Api key, please load it in OPENAI_API_KEY env variable, or alternatively in 'api-key.txt' file.", file=sys.stderr)
        exit(0)
else: print('Loaded api key from environment variable.',file=sys.stderr)

check_api_key_validity(api_key)

s=open(args.source,"rb")

if args.allfile: # Feed the whole description, without parsing it. This is the equivalent of using the ChatGPT web interface.
        l=s.read()
        l=l[:-1]
        l=l.decode()
        l="Write the raw valid %s code for this, ready to be executed, please include comments on each function:\n\n %s" % (args.language,l)
        code=call_AI_chatGPT(l)
        # Remove ChatGPTs code blocks.
        c=""
        for i in code.splitlines():
            if i.find('```')<0:
                c+=i+'\n'
        print(c)
else: # Feed the description line by line

    #Those are valid only for python
    keywords=['#','assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'lambda', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
    line=0
    for l in s.readlines():
        l=l[:-1]
        l=l.decode()
        tabs=l[:l.find(l.strip())]
        code=""
        #keyword line
        for k in keywords:
            if (l.strip().find(k)==0):
                code=l
        #convert line
        if code=="" and len(l)>2:
            l="Write the raw valid %s code for this, ready to be embedded into another %s code:\n %s" % (args.language,args.language,l)
            code=call_AI_chatGPT(l)
            if len(code.splitlines())>1:
                # Remove ChatGPTs '```'
                c=""
                for i in code.splitlines():
                    if i.find('```')<0:
                        c+=i+'\n'
                code=c
        print(tabs+code)
        line+=1
