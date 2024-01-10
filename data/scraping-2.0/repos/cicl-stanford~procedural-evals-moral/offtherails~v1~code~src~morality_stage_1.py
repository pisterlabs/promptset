import random
import csv
import tqdm
import os
import argparse
import ast
import os
import json
import uuid 

#from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# from crfm import crfmChatLLM

from utils import push_data, get_num_items, get_vars_from_out, get_llm

DATA_DIR = '../../data'
PROMPT_DIR = '../prompt_instructions'
SEVERITY_LEVELS = ['Mild', 'Extreme']
SAMPLES_DIR = '../new_prompt_instructions'

# Map story items to tags
STORY_TAGS = json.load(open('story_tags.json'))


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='openai/gpt-4-0613', help='model name')
parser.add_argument('--temperature', type=float, default=0.3, help='temperature')
parser.add_argument('--max_tokens', type=int, default=2000, help='max tokens')
# change num completions to 10
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')
parser.add_argument('--num_shots', type=int, default=3, help='number of shots')
parser.add_argument('--num_stories', type=int, default=2, help='number of stories to generate')
parser.add_argument('--verbose', type=bool, default=False, help='verbose')
parser.add_argument('--api', type=str, default='azure', help='which api to use')

"""
8x1 with evitable vs. inevitable and action vs. prevention for both CC and CoC. 
"""
def gen_conditions(conditions, vars):
    # Means is CC, Side effect is CoC
    for intent in ['CC', 'CoC']:
        intent_phrase = '\"As a means to\"' if intent == 'CC' else '\"As a side effect\"' 
        background = " ".join([vars['Context'], vars[f'Situation {intent}']])
        intro = " ".join([background, vars[f'{intent_phrase} {intent}']])

        # (1) Evitable, Action
        condition = " ".join([intro, vars[f'Evitable Action {intent}'], vars[f'Action {intent}']]) 
        conditions.append(condition)

        # (2) Inevitable, Action
        condition = " ".join([intro, vars[f'External Cause {intent}'], 
                    vars[f'Inevitable Action {intent}'], vars[f'Action {intent}']])
        conditions.append(condition)

        # (3) Evitable, Prevention
        condition = " ".join([intro, vars[f'Other Cause {intent}'], vars[f'Evitable Prevention {intent}'], 
                    vars[f'Prevention {intent}']]) 
        conditions.append(condition)

        # (4) Inevitable, Prevention
        condition = " ".join([background, vars[f'External Cause {intent}'], 
                    vars[f'{intent_phrase} {intent}'], vars[f'Other Cause {intent}'], 
                    vars[f'Inevitable Prevention {intent}'], vars[f'Prevention {intent}']])
        conditions.append(condition)
    return conditions

def get_line(line_indx, filename):
    return open(filename, 'r').readlines()[line_indx]

# Replace harm, good, ext_cause, other_cause in system message
def replace_scenario_vars(line_indx, msg, harm_type, good_type):
    for condition in ['CC', 'CoC']:
        line = get_line(line_indx, f'{SAMPLES_DIR}/{condition.lower()}_stage_1_severe.txt')
        mild_harm, extreme_harm, mild_good, extreme_good = line.strip().split(';')
        harm = mild_harm if harm_type == 'Mild' else extreme_harm
        good = mild_good if good_type == 'Mild' else extreme_good

        msg = msg.replace(f"[Harm {condition}]", harm)
        msg = msg.replace(f"[Good {condition}]", good)

        line = get_line(line_indx, f'{SAMPLES_DIR}/{condition}_causes.txt')
        other_cause_mild, ext_cause_mild, other_cause_extreme, ext_cause_extreme = line.strip().split(';')
        other_cause = other_cause_mild if harm_type == 'Mild' else other_cause_extreme
        ext_cause = ext_cause_mild if harm_type == 'Mild' else ext_cause_extreme

        msg = msg.replace(f"[Other Cause {condition}]", other_cause)
        msg = msg.replace(f"[External Cause {condition}]", ext_cause)
    return msg

def get_human_message(prompt_file, line_indx, harm_type, good_type):
    with(open(f'{PROMPT_DIR}/{prompt_file}.txt', 'r')) as f:
        msg = f.read().strip()

    msg = msg.replace("[Agent]", get_line(line_indx,'background/names.txt'))
    msg = msg.replace("[profession]", get_line(line_indx, 'background/professions.txt').lower())

    # For CC and CoC
    msg = replace_scenario_vars(line_indx, msg, harm_type, good_type)

    # Can randomly sample shots when stable
    msg += open('template_shot.txt', 'r').read()

    return msg

def gen_chat(args):
    response_template = "Here is the story:\n"
    for tag in STORY_TAGS:
        response_template += f"{tag}: {STORY_TAGS[tag]}\n"

    llm = get_llm(args)
    
    template_var = [tag for tag in STORY_TAGS.values()]
    story_file = f'{DATA_DIR}/morality_stage_1_new.csv'

    prompt_tokens_used = 0
    completion_tokens_used = 0

    # Run loop with n stories, increase by num_completions
    for story_indx in tqdm.tqdm(range(0, args.num_stories, args.num_completions)):
        # story_indx = 1 # random.randint(0, len(lines) - 1)
        for harm_type in SEVERITY_LEVELS: 
            for good_type in SEVERITY_LEVELS:

                instruction_text = get_human_message('morality_stage_1', story_indx, harm_type, good_type)

                system_message = SystemMessage(content=instruction_text)
                human_message= HumanMessage(content='Generate a story')
            
                # Read examples from csv file every iteration to add generated samples to the pool of seed examples
                if args.verbose:
                    print(f"Reading examples from {story_file} with {get_num_items(story_file)} existing examples")

                # Read a few examples from the csv file
                examples = []
                # TODO - add this back in once stable 
                # with open(story_file, 'r') as f:
                #     for line in f.readlines():
                #         if ';' not in line:
                #             continue
                #         params = line.split(';')
                #         example = {k: params[v].strip() for v, k in enumerate(template_var)} 
                #         examples.append(example)
                # random.shuffle(examples)

                # re-give prompt after 1st story

                # 2-shots by default	
                messages = [system_message]	
                for i in range(args.num_shots):	
                    if i == len(examples):
                        break
                    messages.append(human_message)
                    messages.append(AIMessage(content=response_template.format(**examples[i])))	

                responses = llm.generate([messages], stop=["System:"])
            
                for g, generation in enumerate(responses.generations[0]):
                    if args.verbose:
                        print(f"------ Generated Story {n_story+g} ------")
                        print(generation.text)
                        print("------------ Fin --------------")
            
                    vars = get_vars_from_out(generation.text)
                
                    # Give unique story ID to cross-reference later
                    story_id = uuid.uuid1().hex

                    # Stitch together a story for each condition
                    conditions = gen_conditions([story_id], vars)
                    
                    # For 8x4 table 
                    with open(f'{DATA_DIR}/{harm_type.lower()}_harm_{good_type.lower()}_good.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile, delimiter=';')
                        writer.writerow(conditions)
                # push to github
                # push_data(DATA_DIR, REPO_URL)
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    gen_chat(args)
