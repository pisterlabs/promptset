#!/usr/bin/env python3
#coding: utf-8

import openai
from openai import OpenAI

import json
import tiktoken
import random

# path to file with authentication key
with open('apikey.txt') as infile:
    apikey = infile.read()

client = OpenAI(api_key=apikey)

# The model shoould try to follow this sort-of meta-instruction
system_message = "You are an author of short stories."

# This is the limit of the model
model_max_tokens = 2048

# How many tokens to generate max
max_tokens = 500

# Model identifier
model = "gpt-3.5-turbo"

def generate_with_openai(messages):

    # Debug output
    # print('MESSAGES:', *messages ,sep='\n')

    # https://platform.openai.com/docs/guides/chat/introduction
    ok = False
    while not ok:
        try:
            response = client.chat.completions.create(
                model = model,
                messages = messages,  # this one only for chat
                max_tokens = max_tokens,
                temperature = 1,
                top_p = 1,
                stop = [], # can be e.g. stop = ['\n']
                presence_penalty = 0,
                frequency_penalty = 0,
                logit_bias = {},
                user = "pribehy",
                )
            ok = True
        except openai.BadRequestError:
            # assume this is because max length is exceeded
            # keep the system message, the prompt and the story title
            # keep removing from the third message
            # TODO do this in a more clever way!
            # explicitly check number of tokens and cut it!
            print(openai.InvalidRequestError)
            messages.pop(3)
    
    result = response.choices[0].message.content

    if result == '':
        # end of text
        print('Nothing was generated, maybe the model assumes that this is a good ending and there is nothing more to add.')
    
    return result

def append_message_user(messages, message):
    messages.append({"role": "user", "content": message})

def append_message_assistant(messages, message):
    messages.append({"role": "assistant", "content": message})


if __name__=="__main__":
    nouns = list()
    with open('nouns.txt') as infile:
        for line in infile:
            nouns.append(line.strip())

    messages = [
        {"role": "system", "content": system_message},
    ]
    
    # slovo = input("   první slovo: ")
    slovo = random.choice(nouns)
    append_message_user(messages, f"Vygeneruj název příběhu, ve kterém se vyskytne {slovo}.")
    title = generate_with_openai(messages)
    print(title)
    append_message_assistant(messages, title)
    
    append_message_user(messages, "Vygeneruj první větu příběhu.")

    while True:
        sentence = generate_with_openai(messages)
        print(sentence)
        append_message_assistant(messages, sentence)
        
        # slovo=input("   další slovo: ")
        slovo = random.choice(nouns)
        append_message_user(messages, f"Vygeneruj další větu příběhu, ve které se vyskytne {slovo}.")

        # "Vygeneruj popisek obrázku, ilustrujícího poslední větu, v angličtině."


