import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import json
from utils import consistent
from greenery import parse
from Levenshtein import distance
from collections import defaultdict
import re
import time
from tqdm import tqdm
from copy import deepcopy

MODEL = "gpt-3.5-turbo-instruct"

prompt_prefix = """import re
'''
Regular expression which matches
- az8az
- ab8ah
- ab88ah
- ab88888ah
- az8888ah
- ab8888888ah
- ab888888888888888ah
and does not match
- azz8az
- abah
- ab8888889888ah
'''
rx = re.compile(r'[a-z]{2}8{1,}[a-z]{2}')

import re
'''
Regular expression which matches
- AAAAAAC
- zAAAC
and does not match
- zAAZC
'''
rx = re.compile(r'[A-Za-z]{1,}A{3}C')

import re
'''
Regular expression which matches
- ahfEIF462
- efERYT8520
- tuysLMEG247
and does not match
- eAB23
- dxbjtPWCNG60935
'''
rx = re.compile(r'[a-z]{2,4}[A-Z]{3,4}[0-9]{3,4}')

import re
'''
Regular expression which matches
- 345GGG
- dGGGG
- DGGG
- A8dGGG
and does not match
- A4d7EGGG
'''
rx = re.compile(r'([0-9]|[A-Za-z]){1,4}G{3,4}')

import re
'''
Regular expression which matches
- dIell5768
- aErll34
- ErBll343457485
and does not match
- DIRlll4345
'''

rx = re.compile(r'[A-Za-z]{3}l{2}[0-9]{2,}')

"""

def extract_program(decoded):
    program = deepcopy(decoded)
    if len(program) == 0:
        return program
    if program[0] == '^':
        program = program[1:]
    if program[-1] == '$':
        program = program[:-1]
    return program

def run_llm_eval(data, completed_interactions, interactions_save_file, keys_save_file):
    usage_stats = defaultdict(int)

    logs = []
    completed_keys = list()
    for d in tqdm(data):
        true_program = d["program"]
        
        if true_program in completed_interactions:
            continue

        spec = d["interaction"][0]["context"]
        user = d["user"]

        try:
            true_program_parsed = parse(true_program.replace('\\', ''))
            print(true_program)
        except:
            continue

        for i in range(1, len(spec) + 1):
            pos_examples = [f"- {x[0]}" for x in spec[:i] if x[1] == "+"]
            pos_examples_str = '\n'.join(pos_examples)
            neg_examples = [f"- {x[0]}" for x in spec[:i] if x[1] == "-"]
            neg_examples_str = 'and does not match\n' + '\n'.join(neg_examples) + '\n' if len(neg_examples) > 0 else ""
            prompt = prompt_prefix + f"import re\n'''\nRegular expression which matches\n{pos_examples_str}\n{neg_examples_str}'''\nrx = re.compile(r'"
            try:
                response = openai.Completion.create(
                    model=MODEL,
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.7,
                    top_p=0.9,
                    n=10,
                    stop=["')", "',"],
                    logprobs=5,
                )
            except openai.error.RateLimitError:
                print("Rate limiting")
                return logs, usage_stats
            
            for k, v in response['usage'].items():
                usage_stats[k] += v

            programs = [extract_program(x['text']) for x in response['choices']]
            consistent_programs = [consistent(p, [x for x in spec[:i]]) for p in programs]
            scores = [sum(x['logprobs']['token_logprobs']) for x in response['choices']]

            valid_guesses = []
            for p, s in zip(programs, scores):
                try:
                    re.compile(p)
                    valid_guesses.append((p, s))
                except:
                    pass
            
            guesses = sorted(valid_guesses, key=lambda x: x[1], reverse=True)

            top_1_success = False
            top_10_success = False
            edit_distance_1 = False
            edit_distance_3 = False
            edit_distance_5 = False

            if len(guesses) > 0:
                try:
                    top_1_success = parse(guesses[0][0]).equivalent(true_program_parsed)
                except:
                    top_1_success = guesses[0][0] == true_program
                
                edit_distance = distance(guesses[0][0], true_program)
                edit_distance_1 = edit_distance <= 1 or top_1_success
                edit_distance_3 = edit_distance <= 3 or top_1_success
                edit_distance_5 = edit_distance <= 5 or top_1_success
            
                
                for p, s in guesses[:10]:
                    try:
                        top_10_success = parse(p).equivalent(true_program_parsed)
                        if top_10_success:
                            break
                    except:
                        top_10_success = p == true_program
                        if top_10_success:
                            break

            logs.append({
                # "interaction_key": interaction_key,
                "program": true_program,
                "spec": spec[:i],
                "user": user,
                "listener": MODEL,
                "top_1_success": top_1_success,
                "top_10_success": top_10_success,
                "turn": i,
                "edit_distance<=1": edit_distance_1,
                "edit_distance<=3": edit_distance_3,
                "edit_distance<=5": edit_distance_5,
                "guessed_programs": programs,
                "consistent_programs": consistent_programs,
                "response": response
            })
        
        with open(interactions_save_file, 'w') as f:
            json.dump(logs, f, indent=4)

    return logs, usage_stats

import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

logs, usage_stats = run_llm_eval(data, {}, sys.argv[2])