import os
import openai
import requests

# openai.organization = "org-CwEqJy6peUJwJURd46VnFBPF"
with open('token.txt') as f:
    lines = f.readlines()
openai.api_key = lines[2].strip()


def emojiTrans(input):
    texts = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Translate the phrase into only unicode emoji: {input}",
        temperature=0.3,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return texts


def generatePhrase():
    phrase = openai.Completion.create(model="text-davinci-003",
                                      prompt="Generate a 2-7 word scenario",
                                      temperature=0.6,
                                      max_tokens=80,
                                      top_p=1,
                                      frequency_penalty=1,
                                      presence_penalty=1
                                      )
    return phrase


def generateHint(input):
    hint = openai.Completion.create(model="text-curie-001",
                                    prompt=f"Rephrase this in the same amount of words: {input}",
                                    temperature=0.7,
                                    max_tokens=256,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0)
    return hint


def parseJSON(jsonFile, opCode):
    if (opCode == 0):
        return ((jsonFile.choices)[0].text).strip(".!@#$%^&&*()")
    if (opCode == 1):
        return ((jsonFile.choices)[0].text).strip().rstrip(",")
