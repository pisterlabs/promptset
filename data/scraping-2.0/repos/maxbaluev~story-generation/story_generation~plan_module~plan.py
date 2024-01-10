import argparse
import csv
from enum import auto
import os
from copy import deepcopy
import pickle
from collections import defaultdict
import multiprocessing as mp
import random
import string
import logging
from time import sleep

import torch
import Levenshtein
import numpy as np
from transformers import AutoTokenizer
import openai
from scipy.special import softmax

from story_generation.edit_module.entity import *
from story_generation.rewrite_module.heuristics import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3_SEP, GPT3_END
from story_generation.common.controller.controller_util import add_controller_args, load_controller
from story_generation.common.controller.loaders.alignment_loader import create_prefix_completion
from story_generation.common.data.split_paragraphs import *

def gen_logit_bias(tokenizer, banned_words_string='', bias=-30, bias_common_tokens=True):
    name_bias_words = ['protagonist', 'Protagonist', 'PROTAGONIST', 'unnamed', 'Unnamed', 'UNNAMED', 'unknown', 'Unknown', 'UNKNOWN', 'None', 'none', 'None', 'Mr', 'Ms', 'Mrs', 'Dr', 'TBA', 'TBD', 'N/A'] # technically no ' can filter out some reasonable names, but it's not a big deal and prevents some bad cases
    banned_name_words = name_bias_words + ['\'', '_', '"', '#', 'redacted', 'mother', 'father', 'parents', 'gram', 'grand', 'name', 'appearance', 'occupation', 'age', 'gender', 'sex', 'role', 'profession', 'job', 'friend', 'kill', 'dead'] #  sometimes it'll find weird ascii chars to replace these if they're banned via logit bias
    name_logit_bias = get_repetition_logit_bias(tokenizer, (banned_words_string + ' ' + ' '.join(banned_name_words)).strip(), bias=bias, bias_common_tokens=bias_common_tokens)
    # name_logit_bias[198] = -5 # also penalize newline, although we want it eventually eventually

    return name_logit_bias

    
def parse_character_portrait(character_portrait):
    chars = {}
    for character in character_portrait.split("Full Name:"):
        if character.strip() == '':
            continue
        name = character.split("\n\n")[0].strip()
        description = character.split("\n\nCharacter Portrait:")[1].strip()
        if "\n\n" in description:
            description = description.split("\n\n")[0].strip()
        if description.startswith(name):
            description = description[len(name):].strip()
        if description.startswith('is'):
            description = description[len('is'):].strip()
        chars[name] = Entity(name, description=name + ' is ' + description[0].lower() + description[1:], is_character=True)
    return chars

def generate_initial_entity_strings(premise, setting, tokenizer, num_entities=3):
    print('Premise: ' + premise)
    print('Setting: ' + setting) 
    initial_characters_prompt = "Premise: " + premise.strip() + '\n\n' + 'Setting: ' + setting.strip() + '\n\nList the Full Names(Name and Surname), Character Portrait(2 or 3 sentences) of all major characters. Each character must be a person in the singular.\n\n1.\n\nFull Name:'
    name_logit_bias=gen_logit_bias(tokenizer)
    

    characters_string = gpt3_completion(initial_characters_prompt, logit_bias=name_logit_bias)
    characters = parse_character_portrait(characters_string)
    print('INITIAL CHARACTERS:', characters)
    
    # Generate new characters if we don't have enough
    add_characters_prompt = None
    if(len(characters) < num_entities):
        print("Not enough characters found in prompt. Generate more.")
        for i in range(num_entities - len(characters)):
            add_characters_prompt = initial_characters_prompt + ' ' + characters_string + '\n\n' + str(len(characters) + 1) + '.\n\nFull Name:'            
            add_characters_string = gpt3_completion(add_characters_prompt, logit_bias=name_logit_bias, freq_pen=1)
            new_characters = parse_character_portrait(add_characters_string)
            # merge new characters into characters
            for name in new_characters:
                if name not in characters:
                    characters[name] = new_characters[name]
            print('NEW CHARACTERS:', new_characters)
    
    # Sometimes gpt generate more characters that we need
    if(len(characters) > num_entities):
        print("Too many characters found in prompt. Truncate.")
        characters = {k: characters[k] for k in list(characters)[:num_entities]}
    
    infer_attributes_string = premise.strip() + '\n\n' + setting.strip() + '\n\n' + '\n\n'.join([ent.description for ent in characters.values()])

    # Old version fix
    character_strings = characters
    if add_characters_prompt is not None:
        characters = add_characters_prompt
    else:
        characters = initial_characters_prompt
    
        
    return  characters, character_strings, infer_attributes_string


def generate_outline(premise, setting, characters, character_strings, instruct_model, generation_max_length, max_sections=5, fixed_outline_length=-1, outline_levels=1, model_string='text-davinci-002'):
    premise_setting_chars = "Premise: " + premise.strip() + '\n\n' + 'Setting: ' + setting.strip() + '\n\n' + 'Characters: ' + characters.strip()

    if fixed_outline_length > 0:
        outline_prompt = premise_setting_chars + '\n\n\n\nOutline the ' + str(fixed_outline_length) + ' main plot points of the story.\n\n1.'
    else:
        outline_prompt = premise_setting_chars + '\n\n\n\nOutline the main plot points of the story.\n\n1.'
    found_acceptable_outline = False
    for i in range(5):
        # bias against repeating the tokens in the prompt, except for the character names themselves
        outline_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, outline_prompt, -2**(i+1))
        name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
        for tok in name_tokens:
            if tok in outline_logit_bias:
                del outline_logit_bias[tok]
        outlines = instruct_model([outline_prompt], logit_bias=outline_logit_bias, generation_max_length=generation_max_length, num_completions=5, model_string=model_string)
        for outline in outlines:
            if fixed_outline_length > 0:
                if str(fixed_outline_length) + '.' not in outline or str(fixed_outline_length+1) + '.' in outline: # looking for exactly this length
                    continue
            if len(split_list('1.' + outline)) < 3: # failure
                continue
            if '2.' not in outline or '3.' not in outline: # properly formatted list and contains at least 3 items
                continue
            if str(max_sections) + '.' in outline: # number of sections in outline exceeds maximum
                continue
            if calculate_repetition_length_penalty(outline, [setting, characters], is_outline=True) > 0: # it's fine if some of the premise is repeated e.g. in the early parts
                continue
            if len(instruct_model.tokenizer.encode(outline)) < generation_max_length: # ideally, terminate because the outline is done, not because it was too long
                found_acceptable_outline = True
                break
        if found_acceptable_outline:
            break
    if not found_acceptable_outline:
        logging.warning('Warning: didn\'t find acceptable outline')
        raise ValueError
    outline = ('1.' + outline).strip()
    logging.log(23, outline)
    if outline_levels > 1:
        all_detailed_outlines = []
        assert outline_levels == 2 # in principle could support more
        for outline_idx, outline_piece in enumerate(split_list(outline)):
            found_acceptable_outline = False
            for i in range(5):
                detailed_outline_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, outline_prompt + ' ' + ' '.join([op for op in split_list(outline)]), -2**(i+1))
                name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
                for tok in name_tokens:
                    if tok in outline_logit_bias:
                        del outline_logit_bias[tok]
                detailed_outlines = instruct_model([premise_setting_chars + '\n\nOutline:\n\n' + '\n\n'.join([op for op in split_list(outline)[:outline_idx]]) + '\n\nList the minor events in the next part of the story, in which ' + outline_piece.strip() + '\n\n1.'], logit_bias=detailed_outline_logit_bias, generation_max_length=generation_max_length, num_completions=5, model_string=model_string)
                for detailed_outline in detailed_outlines:
                    if fixed_outline_length > 0:
                        if str(fixed_outline_length) + '.' not in detailed_outline or str(fixed_outline_length+1) + '.' in detailed_outline: # looking for exactly this length
                            continue
                    if len(split_list('1.' + detailed_outline)) < 3: # failure
                        continue
                    if '2.' not in detailed_outline or '3.' not in detailed_outline: # properly formatted list and contains at least 3 items
                        continue
                    if str(max_sections) + '.' in detailed_outline: # number of sections in outline exceeds maximum
                        continue
                    if calculate_repetition_length_penalty(detailed_outline, [setting, characters, outline], is_outline=True) > 0: # it's fine if some of the premise is repeated e.g. in the early parts
                        continue
                    if len(instruct_model.tokenizer.encode(detailed_outline)) < generation_max_length: # ideally, terminate because the outline is done, not because it was too long
                        found_acceptable_outline = True
                        break
                if found_acceptable_outline:
                    break
            if not found_acceptable_outline:
                logging.log(23, 'Warning: didn\'t find acceptable outline')
                raise ValueError
            all_detailed_outlines.append('1.' + detailed_outline)
        outline = (outline, all_detailed_outlines)
    return outline


def load_plan_info(plan_file):
    with open(plan_file, 'rb') as f:
        save_info = pickle.load(f)
    return save_info


def generate_plan_info(args, instruct_model, include_outline=True, model_string='text-davinci-002'):
    tokenizer = instruct_model.tokenizer

    premise = args.premise.strip()
    logging.log(25, 'Premise: ' + premise)

    setting_prompt = "Premise: " + premise.strip() + '\n\nDescribe the setting of the story in one sentense.\n\nThe story is set in'
    logit_bias = gen_logit_bias(tokenizer, banned_words_string=setting_prompt)
    setting = 'The story is set in ' + gpt3_completion(setting_prompt, freq_pen=1, logit_bias=logit_bias)
    characters, character_strings, infer_attributes_string = generate_initial_entity_strings(premise, setting, tokenizer)

    if not include_outline:
        outline = None
    outline_max_tokens = 128
    outline = generate_outline(premise, setting, characters, character_strings, instruct_model, outline_max_tokens, fixed_outline_length=args.fixed_outline_length, outline_levels=args.outline_levels)

    # assume gpt3 was smart enough to number them when prompted
    if type(outline) == tuple:
        outline_sections = sum([split_list(op) for op in outline[1]], [])
    else:
        outline_sections = split_list(outline)

    logging.log(25, 'Outline: ' + str(outline))

    # do the attribute inference after outlines are generated, since it can be expensive
    if not args.no_attributes and not args.no_editor and not args.no_planner:
        for entity in character_strings.values():
            entity.infer_attributes(infer_attributes_string, instruct_model, other_names=[name for name in character_strings.keys() if name != entity.name])
        complete_mutual_relations(character_strings, instruct_model)         

    save_info = {'premise': premise,
        'setting': setting,
        'characters': characters,
        'character_strings': character_strings,
        'outline': outline,
        'outline_sections': outline_sections,
        'infer_attributes_string': infer_attributes_string}
    return save_info

def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, logit_bias=None, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
    kwargs = dict(engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                logit_bias=logit_bias,
                stop=stop)
    while True:
        try:
            response = openai.Completion.create(**{k: v for k, v in kwargs.items() if v is not None})
            text = response['choices'][0]['text'].strip()
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)