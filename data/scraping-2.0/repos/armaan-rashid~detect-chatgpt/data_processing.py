"""
This file contains some basic data processing utility functions. 
Can be run as a script to either repair unfinished data, merge data
or load data from files into the main ChatGPT script. 
Some of these functions are from Mitchell et al.'s 
detectGPT. Their original code can be found here:
https://github.com/eric-mitchell/detect-gpt
"""

import pandas as pd
from argparse import ArgumentParser
import openai
import os
import torch
from transformers import AutoTokenizer

# housekeeping some global vars
USAGE = 0
FAILSTRING = 'Failed response.'
SYSTEM = {'role': 'system', 'content': 'You are a helpful assistant.'}   # a default system msg to use for all prompts 
CONTINUE = {'role': 'user', 'content': 'Please, continue.'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(filename, k=None, tokenizer=None):
    """
    Load k examples of data from file into dict format.
    Expects that the dfs loaded in has 'original, sampled'
    columns and ignores other columns.
    """
    df = pd.read_csv(filename)
    assert 'original' in df.columns and 'sampled' in df.columns, 'files need to have original and sampled cols'
    print(f'Loading data from {filename}.')
    conv = {'original': df['original'].values.tolist(),
            'sampled': df['sampled'].values.tolist()}
    k = min(len(conv['original']), k) if k != 0 else len(conv['original'])
    conv['original'] = conv['original'][:k]
    conv['sampled'] = conv['sampled'][:k]
    if tokenizer:
        print(f'Verifying that all passages have length less than {tokenizer.model_max_length} tokens.')
        try: tokenizer.to(DEVICE)
        except: pass
        conv['original'] = [truncate_tokens(example) for example in conv['original']]
        conv['sampled'] = [truncate_tokens(example) for example in conv['sampled']]
    return conv

def truncate_tokens(string, tokenizer):
    """
    Truncate a string to be the max length of a tokenizer.
    """    
    tokenized = tokenizer.encode(string)
    if len(tokenized) > tokenizer.model_max_length:
        print(f'Truncating an example because it uses too many ({len(tokenized)}) tokens')
        return tokenizer.decode(tokenized[:tokenizer.model_max_length])
    return string

def truncate_dataframe(df: pd.DataFrame, tokenizer):
    """
    Truncate tokens for all the entries in a df full of strings.
    """
    return df.applymap(lambda item: truncate_tokens(item, tokenizer))

def concat_cols(row, cols):
    string = ''
    for col in cols:
        string += row[col] + ' '
    return string.strip()


def match_lengths(data: pd.DataFrame, col1: str, col2: str):
    """
    Given a DataFrame of two columns, truncate the 
    original-sampled pairs to roughly match length (i.e. have same
    word count.) 
    """
    for i, row in data.iterrows():
        orig_split = row[col1].split()
        sampled_split = row[col2].split()
        trunc = min(len(orig_split), len(sampled_split))
        row[col1] = ' '.join(orig_split[:trunc])
        row[col2] = ' '.join(sampled_split[:trunc])
    return data

def remove_failed_responses(file):
    """
    Erase the failed responses that ChatGPT couldn't generate.
    """
    df = pd.read_csv(file)
    idxs = [i for i, row in df.iterrows() if FAILSTRING in row['sampled']]
    df.drop(labels=idxs, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(file, index=False)
    print(f'removed {len(idxs)} responses from {file}')

def process_spaces(story: str):
    """Basic processing function, adapted from Mitchell et al."""
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', ' ').replace(
        '\n', ' ').replace(
        '  ', ' ').strip()


def replace_original(correct: pd.DataFrame, incorrect: pd.DataFrame):
    """Emergency function to handle misplaced perturbations."""
    c = len(incorrect.columns) // 2
    n = len(incorrect)
    for i in range(c):
        incorrect[f'o{i+1}'] = correct[f'o{i+1}'][:n]
    return incorrect
    


def repair_dataframe(data: pd.DataFrame, temp: float, min_words=200, prompt_msg=''):
    """
    DESC: Repair dataframe that has incomplete responses from ChatGPT.
    PARAMS:
    data: a dataFrame that has both a 'prompts' and 'responses' column
    chatbot: logged in ChatGPT
    verbose: print chatGPT's responses while querying 
    """
    fail = 0
    count = 0
    for _, row in data.iterrows():
        if row['responses'] == FAILSTRING:
            try: 
                prompt = prompt_msg + row['prompts']
                response = prompt_ChatGPT(prompt, temp, min_words)
                row['responses'] = response
                count += 1
            except:
                print(f'The prompt: {prompt} did not successfully get a response from ChatGPT.\n')
                fail += 1
                continue
    print(f'Successfully got {count} responses from ChatGPT, failed to get {fail} responses.')
    return data

def prompt_ChatGPT(prompt: str, temp: float, min_words=250, postprocess=process_spaces):
    """
    DESC: Self-explanatory, prompts OpenAI API with prompt
    til response length greater than min_words.
    CALLED_BY: generate() funcs
    """
    global USAGE
    msgs = [SYSTEM, {'role': 'user', 'content': prompt}]
    response_len = 0

    while response_len < min_words:
        if response_len != 0:
            msgs.append(CONTINUE)
        r = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=msgs, temperature=temp)
        USAGE += r['usage']['total_tokens']
        msgs.append(r['choices'][0]['message'])
        this_len = len(msgs[-1]['content'].split())
        response_len += this_len
    
    response = ' '.join([msg for msg in msgs if msg['role'] == 'assistant'])
    return postprocess(response)

def merge_human_sampled(original_file, original_cols, sampled_file, sampled_cols, outfile=None):
    """
    DESC: Given files of both original and sampled data,
    merge them into one dataFrame.
    PARAMS: 
    original_file, sampled_file: file of human data, chatGPT data resp.
    original_cols, sampled_cols: list of cols to read in from original_file, sampled_file resp. 
        if there are multiple columns, they're concatenated with a space separating the strings in each.
    outfile: where to write merged data
    RETURNS: dataFrame of merged data
    """
    original = pd.read_csv(original_file)
    sampled = pd.read_csv(sampled_file)
    
    if original_cols is None:
        original_cols = original.columns
    if sampled_cols is None:
        sampled_cols = sampled.columns

    original['original'] = original.apply(lambda row: concat_cols(row, original_cols), axis=1)
    sampled['sampled'] = sampled.apply(lambda row: concat_cols(row, sampled_cols), axis=1)
    df = pd.concat([original['original'], sampled['sampled']], axis=1)
    df = match_lengths(df, 'original', 'sampled')
    if outfile:
        df.to_csv(outfile, index=False)
    return df


def strip_text(file, col, strip_msg):
    df = pd.read_csv(file)
    assert col in df.columns, 'invalid column called for this dataFrame'
    df[col] = df.apply(lambda row: row[col].replace(strip_msg, ''), axis=1)
    df.to_csv(file, index=False)
    print(f'Stripped the text \'{strip_msg}\' from {file} in column {col}')
    




if __name__=='__main__':
    parser = ArgumentParser(prog='process data already retrieved, in different ways')
    parser.add_argument('task', help='what you want to do', choices=['merge', 'repair', 'strip', 'remove', 'truncate'])
    merge = parser.add_argument_group()
    merge.add_argument('--orig_file', help='file with human data')
    merge.add_argument('--orig_cols', help='cols to grab from orig_file', type=str)
    merge.add_argument('--sampled_file', help='file with ChatGPT data')
    merge.add_argument('--sampled_cols', help='cols to grab from data', type=str)
    merge.add_argument('--outfile', help='where to store new merged data')
    repair = parser.add_argument_group()
    repair.add_argument('--repair_file', help='file with data that needs to be repaired')
    repair.add_argument('--temp', help='for ChatGPT prompting', type=float)
    repair.add_argument('--min_words', help='for ChatGPT prompting', type=int)
    repair.add_argument('--prompt_msg', help='message to append to beginning of prompt during repair')
    strip = parser.add_argument_group()
    strip.add_argument('--strip_file', help='file to strip from')
    strip.add_argument('--strip_col', help='col to strip from')
    strip.add_argument('--strip_msg', help='text to strip')
    remove = parser.add_argument_group()
    remove.add_argument('--remove_files', help='files with rows to remove', nargs='*')
    truncate = parser.add_argument_group()
    truncate.add_argument('--trunc_files', help='files you want to truncate with a tokenizer', nargs='*')
    truncate.add_argument('--tokenizer', help='what pretrained tokenizer you want to use for truncation')

    parser.add_argument('-v', '--verbose', action='store_true', help='print while doing stuff')
    args = parser.parse_args()

    if args.task == 'merge':
        assert args.orig_file and args.sampled_file, 'need to have files to merge!'
        orig_cols = args.orig_cols.split(', ')
        sampled_cols = args.sampled_cols.split(', ')
        merged = merge_human_sampled(args.orig_file, orig_cols, args.sampled_file, sampled_cols, args.outfile)
    

    elif args.task == 'repair':
        openai.api_key = os.getenv('OPENAI_API_KEY')
        broken = pd.read_csv(args.repair_file)
        fixed = repair_dataframe(broken, args.temp, args.min_words, args.prompt_msg)
        fixed.to_csv(args.repair_file, index=False)
        print(f'Used {USAGE} tokens in this run.')

    elif args.task == 'strip':
        assert args.strip_file and args.strip_col and args.strip_msg
        strip_text(args.strip_file, args.strip_col, args.strip_msg)

    elif args.task == 'remove':
        for file in args.remove_files:
            remove_failed_responses(file)
    
    elif args.task == 'truncate':
        for file in args.trunc_files:
            perturbed = pd.read_csv(file)
            perturbed = truncate_dataframe(perturbed, AutoTokenizer.from_pretrained(args.tokenizer))
            perturbed.to_csv(file, index=False)
