import os
import openai
from Synthezator import *
import json

def run(prompt):
        synthezator = Synthezator()
        #read promt adjustment from json file
        with open('prompt_adjustment.json') as json_file:
            data = json.load(json_file)
            prompt_adjustment = data['prompt_adjustment']
        prompt = prompt_adjustment + prompt
        anwser = ask(prompt)
        file = synthezator.synthesize(anwser)
        os.system("afplay " + file)
        print("Bot: " + anwser)


def ask(PROMPT, MaxToken=3900, outputs=1): 
        response = openai.Completion.create( 
            model="text-davinci-003", 
            prompt=PROMPT, 
            max_tokens=MaxToken, 
            n=outputs,
            temperature=0.6
        ) 

        return response.choices[0].text  





