import random
import csv
import tqdm
import os
import argparse
import os
from typing import List

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chat_models import AzureChatOpenAI

DATA_DIR = '../../data'
PROMPT_DIR = '../offtherails'

from utils import get_llm, get_vars_from_out

def get_context(name, profession):
    # check if profession is noun
    if profession.strip()[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        profession = f'an {profession.strip()}'
    else:
        profession = f'a {profession.strip()}'
    context = f"{name.strip()}, {profession}, faces a moral dilemma."
    return context




CONDITION = ['CC', 'CoC']


def get_example(condition, rand_item):

    vars = {k: None for k in range(100)}

    if condition == "cc":
        with open(f'{PROMPT_DIR}/cc_stage_2_mild.csv', 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for i, row in enumerate(reader):
                if i == rand_item:
                    for j, elem in enumerate(row):
                        vars[j] = elem.strip()
                    break
        return f"""Context: {vars[0]}
Action Opportunity: {vars[1]}
Harm CC: {vars[2]}
Good CC: {vars[3]}
Preventable Cause CC: {vars[4]}
Non-Preventable Cause CC: {vars[5]}
"As a means to" CC: {vars[6]}
Evitable Action CC: {vars[7]}
Inevitable Action CC: {vars[8]}
Evitable Prevention CC: {vars[9]}
Inevitable Prevention CC: {vars[10]}
Action CC: {vars[11]}
Prevention CC: {vars[12]}"""

    elif condition == "coc":
        with open(f'{PROMPT_DIR}/coc_stage_2_mild.csv', 'r') as f:  
            reader = csv.reader(f, delimiter=';')
            for i, row in enumerate(reader):
                if i == rand_item:
                    for j, elem in enumerate(row):
                        vars[j] = elem.strip()
                    break
        return f"""Context: {vars[0]}
Action Opportunity: {vars[1]}
Harm CoC: {vars[2]}
Good CoC: {vars[3]}
Preventable Cause CoC: {vars[4]}
Non-Preventable Cause CoC: {vars[5]}
"As a side effect" CoC: {vars[6]}
Evitable Action CoC: {vars[7]}
Inevitable Action CoC: {vars[8]}
Evitable Prevention CoC: {vars[9]}
Inevitable Prevention CoC: {vars[10]}
Action CoC: {vars[11]}
Prevention CoC: {vars[12]}"""
                
            
def gen_chat(args, condition):
    llm = get_llm(args)
    
    vars = {k: None for k in range(100)}

    # load names 
    with(open(f'{PROMPT_DIR}/names.txt', 'r')) as f:
        names = f.readlines()

    # load professions
    with(open(f'{PROMPT_DIR}/professions.txt', 'r')) as f: 
        professions = f.readlines()

    
    for i in range(args.start, args.end):

        name = names[i]
        profession = professions[i]

        # load example
        rand_item = 0 #random.randint(0, 1)
        example = get_example(condition, rand_item=rand_item)
    

    
        # messages sent to model 
        messages = []
        if condition == "cc":
            with(open(f'{PROMPT_DIR}/cc_stage_2.txt', 'r')) as f:
                system_prompt = f.read().strip()
            
            with(open(f'{PROMPT_DIR}/cc_stage_1_mild.csv', 'r')) as f:
                reader = csv.reader(f, delimiter=';')
                new_item = list(reader)[i]
     
            human_message_1 = HumanMessage(content=f"""Generate a new completion for this context: 
Context: {get_context(name=name, profession=profession)}
Action Opportunity: {new_item[0]}
Harm CC: {new_item[1]}
Good CC: {new_item[2]}
Preventable Cause CC: {new_item[3]}
Non-Preventable Cause CC: {new_item[4]}""")

        elif condition == "coc":
            with(open(f'{PROMPT_DIR}/coc_stage_2.txt', 'r')) as f:
                system_prompt = f.read().strip()

            with(open(f'{PROMPT_DIR}/coc_stage_1_mild.csv', 'r')) as f:
                reader = csv.reader(f, delimiter=';')
                new_item = list(reader)[i]

            human_message_1 = HumanMessage(content=f"""Generate a new completion for this context: 
Context: {get_context(name=name, profession=profession)}
Action Opportunity: {new_item[0]}
Good CoC: {new_item[1]}
Harm CoC: {new_item[2]}
Preventable Cause CoC: {new_item[3]}
Non-Preventable Cause CoC: {new_item[4]}""")

        system_message = SystemMessage(content=system_prompt)
        human_message_0 = HumanMessage(content=f"Generate a completion.")
        ai_message_0 = AIMessage(content=example)
    


        messages.append(system_message)
        messages.append(human_message_0)
        messages.append(ai_message_0)
        messages.append(human_message_1)

        responses = llm.generate([messages], stop=["System:"])


        for g, generation in enumerate(responses.generations[0]):
            if args.verbose:
                print(f"------ Generated Story ------")
                print(generation.text)
                print("------------ Fin --------------")

  
            new_vars = get_vars_from_out(generation.text)
            vars = [get_context(name=name, profession=profession)] + new_item + new_vars

            breakpoint()
    
            if condition == "cc":
                with open(f'{PROMPT_DIR}/cc_stage_2_mild.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(vars)
            elif condition == "coc":
                with open(f'{PROMPT_DIR}/coc_stage_2_mild.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(vars)
            
            # breakpoint()

           
            

            
            
            


       
    

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=1, help='start index')
parser.add_argument('--end', type=int, default=10, help='end index')
parser.add_argument('--model', type=str, default='openai/gpt-4-0613', help='model name')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
parser.add_argument('--max_tokens', type=int, default=2000, help='max tokens')
# change num completions to 10
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')
parser.add_argument('--num_shots', type=int, default=3, help='number of shots')
parser.add_argument('--num_stories', type=int, default=2, help='number of stories to generate')
parser.add_argument('--verbose', type=bool, default=True, help='verbose')
parser.add_argument('--api', type=str, default='azure', help='which api to use')

if __name__ == "__main__":
    args = parser.parse_args()
    gen_chat(args, condition='coc')