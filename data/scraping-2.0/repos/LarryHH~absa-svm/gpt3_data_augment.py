import os
import sys
import re
import random
import configparser
from typing import Union, Tuple, List
import openai

from dataset_depparse import *
from file_utils import * 

import nltk
nltk.download('words')
nltk.download('webtext')
from nltk.corpus import words, webtext
from wordsegment import load as wordsegment_load, segment
wordsegment_load()

def load_absa_dataset(fp: str) -> List[Sample]:
    data = []
    lines = read_as_list(fp)
    for i in range(len(lines) // 3):
        text = lines[i * 3]
        aspect = lines[i * 3 + 1]
        polarity = int(lines[i * 3 + 2])       
        text_formatted = format_hashstring(text, aspect) 
        data.append(Sample(text_formatted, aspect, polarity))
    return data

def load_gpt3_config(fp):
    config = configparser.ConfigParser(allow_no_value=True)		
    config.read(fp)
    args = config['GPT3']
    args = {
        'api_key': args['api_key'],
        'iters': int(args['iters']),
        'review_type': args['review_type'],
        'sent_label': int(args['sent_label']),
        'n_samples': int(args['n_samples']),
        'aspect': args['aspect'],
        'engine': args['engine'],
        'temperature': float(args['temperature']),
        'max_tokens': int(args['max_tokens']),
        'top_p': float(args['top_p']),
        'frequency_penalty': float(args['frequency_penalty']),
        'presence_penalty': float(args['presence_penalty'])
    }
    return args

def format_hashstring(text: str, aspect: str) -> str:
    # fix malformed hashwords, e.g. '*##', '##*', '*##*', and if aspect has multiple terms, replace ## with aspects
    tokens = text.split(' ')
    idx = [i for i, token in enumerate(tokens) if '##' in token][0]
    hashwords = tokens[idx]
    hashwords = [w if w else "'" for w in hashwords.split("'")]
    tokens[idx] = '##'
    if len(hashwords) > 1:
        tokens = [y for x in tokens for y in ([x] if x != '##' else hashwords)]
    tokens = [t if '##' not in t else '##' for t in tokens]
    hashwords = [w for w in hashwords if w != "'"][0]
    # idx = tokens.index('##')
    aspect_trunc = ''
    # fix truncated aspect
    if len(hashwords) > 2:
        if hashwords.startswith('#'): # "##*"
            aspect_trunc = hashwords.split('##')[1]
            aspect = aspect + aspect_trunc
        elif hashwords.endswith('#'): # "*##"
            aspect_trunc = hashwords.split('##')[0] 
            aspect = aspect_trunc + aspect
        else: # "*##*"
            aspect_trunc = hashwords.split('##') 
            aspect = aspect.join(aspect_trunc)
    # replace '##' with aspects (single or multi)
    aspects = aspect.split(' ')
    tokens = ' '.join([y for x in tokens for y in ([x] if x != '##' else aspects)])
    return tokens

def parse_compounds_and_ticks(sentence):
    output = []
    for word in sentence.split(" "):
        if word not in CORPUS:
            seg = segment(word)
            if len(seg) == 1:
                seg = seg[0]
                if any((match := char) in ',.:;\?!)' for char in word):
                    seg += match
                if any((match := char) in '(' for char in word):
                    seg = match + seg
                if any((match := char) in "'" for char in word):
                    seg = seg[:word.index("'")] + "'" + seg[word.index("'"):]
            output.append([seg]) if isinstance(seg, str) else output.append(seg)
        else:
            output.append([word])
    return ' '.join([word for tokens in output for word in tokens])

def clean_sentences(data: List[Sample], sent_label: int, aspect: str=None, n_samples: int=None) -> List[str]:
  # 2 modes? from file and from samples?
    if aspect:
        sentences = [s.text for s in data if s.polarity == sent_label and s.aspect == aspect]
    else:
        sentences = [s.text for s in data if s.polarity == sent_label]
    if n_samples:
        sentences = sentences[:n_samples]
    # attach punctuation properly. multiple ticks and compound words not handled here
    sentences = [re.sub("( [,.:;\?!)]|[\(] | ['] )", lambda x: x.group(1).strip(), s) for s in sentences]
    return [parse_compounds_and_ticks(s) for s in sentences]

def gpt3_format_header(sent_label: str, review_type: str, aspect: str=None) -> str:
    if aspect:
        sent_map = {
            1: f"Positive {review_type} reviews about {aspect.capitalize()}",
            0: f"Neutral {review_type} reviews about {aspect.capitalize()}",
            -1: f"Negative {review_type} reviews about {aspect.capitalize()}"
        }
    else:
        sent_map = {
            1: f"Positive {review_type} reviews",
            0: f"Neutral {review_type} reviews",
            -1: f"Negative {review_type} reviews"
        }
    try:
        header = '\n\n' + sent_map[sent_label]
    except KeyError as e:
        print(f'ERROR: given label should map to known sentiment: 1 = pos, 0 = net, -1 = neg. Given label: {e}')
        sys.exit(1)
    return header

def gpt3_generate_review(args, data: List[str]):

    header = gpt3_format_header(args['sent_label'], args['review_type'], args['aspect'])
    n_samples = args['n_samples'] if args['n_samples'] < len(data) else len(data) 
    body = random.sample(data, n_samples)
    prompt = f"{header}\n\n1. "
    for i, b in enumerate(body):
        prompt += f"{b}\n\n{i+2}. "
    print(prompt)

    openai.api_key = args['api_key']

    for i in range(args['iters']):
        response = openai.Completion.create(
            engine=args['engine'],
            prompt=prompt,
            temperature=args['temperature'],
            max_tokens=args['max_tokens'],
            top_p=args['top_p'],
            frequency_penalty=args['frequency_penalty'],
            presence_penalty=args['presence_penalty']
        )
        gen = response.choices[0].text
        print(gen)
        # gen_list_id = int(gen.split('\n\n')[-2][0])
        # prompt = f"{prompt}{gen}{gen_list_id}. "
        prompt += gen
    #json.dump(response, open('_gpt3.json', 'w'))
    with open(f"{base_dir}raw_{args['review_type']}_{args['sent_label']}_{args['aspect']}.txt", 'a+') as f:
        f.writelines(prompt)

if __name__ == "__main__":
    CORPUS = set(words.words()).union(set(webtext.words()))

    base_dir = 'datasets/rest/gens/'

    training_path = os.path.join(base_dir, 'train.txt')
    testing_path = os.path.join(base_dir, 'test.txt')
    train_data = load_absa_dataset(training_path)
    test_data = load_absa_dataset(testing_path)

    args = load_gpt3_config('config/config.ini')
    print(args)
    sentences = clean_sentences(train_data, args['sent_label'], args['aspect'], args['n_samples'])
    gpt3_generate_review(args, data=sentences)