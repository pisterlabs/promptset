"""
This file implements the functionality for generating ChatGPT passages.

Each dataset has a LOAD function which loads the human dataset from wherever 
it may be: HuggingFace, local files, etc.

Each dataset also has a GENERATE function which takes in a human dataset
and prompts ChatGPT to generate examples in whatever way is appropriate 
to that dataset: asking a question, asking it to complete text, etc.

When run as a script, main() calls a LOAD function to create prompts
and then a GENERATE function to create responses for a dataset. The GENERATE funcs
call the core ChatGPT interfaces prompt_from_dataframe/prompt_ChatGPT. 

There are lots of options for I/O at multiple stages in the querying process.
Generally, we use .csv files and DataFrames because it's easy. 
"""


# from google.cloud import bigquery
import pandas as pd
import transformers
import random
import os
import openai
from datasets import load_dataset
from torch import cuda
from data_processing import process_spaces, match_lengths
from argparse import ArgumentParser

USAGE = 0
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
FAILSTRING = 'Failed response.'
SYSTEM = {'role': 'system', 'content': 'You are a helpful assistant.'}   # a default system msg to use for all prompts 
CONTINUE = {'role': 'user', 'content': 'Please, continue.'}

def prompt_from_dataframe(data: pd.DataFrame, temp, min_words: int):
    """
    DESC: Query ChatGPT to generate a response for every prompt and
    append these responses to a dataFrame.
    PARAMS:
    data: dataFrame with prompts in it
    chatbot: ChatGPT already logged in
    verbose: print ChatGPT's responses or not
    min_words: min length of valid response from ChatGPT
    RETURNS:
    df: dataFrame with prompts and responses
    """
    count = 0
    fail = 0
    responses = []

    def prompt_ChatGPT(prompt: str, postprocess=process_spaces):
        """
        DESC: Self-explanatory, prompts OpenAI API with prompt
        til response length greater than min_words.
        CALLED_BY: generate() funcs
        """
        global USAGE
        response_len = 0
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
        
        response = ' '.join([msg['content'] for msg in msgs if msg['role'] == 'assistant'])
        return postprocess(response)

    for prompt in data['prompts']:
        try:
            responses.append(prompt_ChatGPT(prompt))
            count += 1
        except:
            fail += 1
            responses.append(FAILSTRING)
            print(f'Failed to get response to \"{prompt}\" from ChatGPT. Moving on to next prompt. Use data_processing ' 
                   'script to repair later.')

    data['responses'] = responses   # add responses to the DF
    print(f'Successfully got {count} responses from ChatGPT at temperature {temp}. Failed to get {fail} responses.')
    return data 
        

# def bigquery_load(sql, outfile):
#     """
#     Pass a SQL query to bigQuery, 
#     save results as JSON in outfile.
#     """
#     client = bigquery.Client()
#     df = client.query(sql).to_dataframe()
#     df.to_json(outfile)
#     print(f"Received {len(df)} examples from BigQuery.")


def load_human_data(file, num_examples):
    """
    Self-explanatory: load n examples of human data, i.e. prompts, from a file.
    """
    df = pd.read_csv(file)
    assert len(df) >= num_examples and num_examples > 1, 'need to choose more than 1 example, or too many examples for file'
    return df.loc[:num_examples]


def xsum_load(infile=None, outfile=None, num_examples=500, preprocess=process_spaces):
    """
    DESC: Download XSum from HuggingFace datasets hub, or load from file.
    PARAMS:
    infile: file where dataset already lives, if applicable
    outfile: file to write human data to if applicable
    num_examples: num to take from HuggingFace dataset
    preprocess: function for preprocessing examples
    RETURNS: DataFrame of human XSum examples
    """
    if infile:
        return load_human_data(infile, num_examples)
    xsum_dict = load_dataset('xsum')
    xsum = xsum_dict['train']
    articles = [preprocess(xsum[idx]['document']) for idx in random.sample(range(len(xsum)), num_examples)]
    df = pd.DataFrame({'articles': articles})
    if outfile:
        df.to_csv(outfile, index=False)
    return df


def xsum_generate(xsum: pd.DataFrame, temp: float, tokens=30, prompt_msg='', min_words=250, outfile=None):
    """
    DESC: Truncate the news articles in the XSum data and prompt
    ChatGPT. This function is different than the functions for other datasets
    because we're calling a tokenizer to cut off the prompt at 
    the length specified by tokens, whereas the other datasets have a natural 
    notion of prompt. Part of this function adapted from Mitchell et al.'s
    original ChatGPT implementation @ https://github.com/eric-mitchell/detect-gpt
    PARAMS: 
    xsum: DataFrame of XSum news articles (needs 'articles' column)
    tokens: number of tokens from article to prompt ChatGPT with
    prompt_msg: add'l message to prompt ChatGPT with BEFORE news article
    min_words: min length of valid response from ChatGPT
    retain: write prompts to outfile if True
    outfile: file to write prompts/responses to
    RETURNS: DataFrame of generated XSum examples
    """
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    try:    # this needs to be try/except for compatibility with different versions of datasets API
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        tokenizer.pad_token_id = [tokenizer.eos_token_id]
    tokenized = tokenizer(xsum['articles'].values.tolist(), return_tensors="pt", padding=True).to(DEVICE)
    tokenized = {key: value[:, :tokens] for key, value in tokenized.items()}

    prompts = tokenizer.batch_decode(tokenized['input_ids'], skip_special_tokens=True)
    xsum['prompts'] = [prompt_msg + prompt for prompt in prompts]
    xsum = prompt_from_dataframe(xsum, temp, min_words=min_words)
    if outfile:
        xsum[['prompts', 'responses']].to_csv(outfile, index=False)
    return xsum



def squad_load(infile=None, outfile=None, num_examples=500, preprocess=process_spaces):
    """
    DESC: Download SQuAD from HuggingFace hub, or from file.
    Sample num_examples if downloading.
    PARAMS:
    infile: file with already loaded data
    outfile: file to write human data
    num_examples: number to sample if downloading from HuggingFace
    preprocess: preprocessor function to apply to each example
    RETURNS:
    dataFrame with contexts, questions, and answers
    """
    if infile:
        return load_human_data(infile, num_examples)
    squad_dict = load_dataset("squad")
    squad = squad_dict['train']
    idxs = random.sample(range(len(squad)), num_examples)
    contexts = [preprocess(squad[idx]['context']) for idx in idxs]
    questions = [preprocess(squad[idx]['question']) for idx in idxs]
    answers = [preprocess(squad[idx]['answers']['text'][0]) for idx in idxs]
    df = pd.DataFrame({'contexts': contexts, 'questions': questions, 'answers': answers})
    if outfile:
        df.to_csv(outfile, index=False)
    return df



def squad_generate(squad: pd.DataFrame, temp: float, min_words: int, outfile=None):
    """
    DESC: Given a dataFrame of SQuAD q's, a's, contexts, prepare data
    to feed in as prompts to ChatGPT. Write to outfile if provided.
    PARAMS:
    squad: DataFrame of squad examples (must have contexts and questions cols)
    prompt_msg: msg to prompt chatGPT with in addition to questions
    min_words: min valid length of chatGPT response
    retain: write prompts to outfile
    outfile: file to write prompts/responses
    RETURNS:
    squad: DataFrame with chatGPT responses
    """
    squad['prompts'] = squad.apply(lambda row: row['contexts'] + ' ' + row['questions'], axis=1)
    squad = prompt_from_dataframe(squad, temp, min_words=min_words)
    if outfile:
        squad[['prompts', 'responses']].to_csv(outfile, index=False)
    return squad


def wp_load(infile: str, num_examples, outfile=None, load=False):
    """
    DESC: Another loading function, this time for Reddit WritingPrompts.
    Some quirks because this dataset is stored in large files.
    PARAMS:
    infile: this could be ONE infile if args.load is false, TWO if it's true. If it's two,
        then it's assumed the wp_source file is passed in first, then matching wp_target file.
    num_examples: num_examples to load 
    outfile: outfile to save data to if necessary
    load: True if args.load is true, false otherwise
    RETURNS: two column DataFrame, one of prompts and the other of stories
    """
    if not load:   # if args.load is true, assume infile is already-prepped csv
        return load_human_data(infile, num_examples)
    split = infile.find(' ')
    source, target = infile[:split], infile[split+1:]

    def remove_prompt_tag(string):   # implementation of this utility from Mitchell et al.
        return string.replace('[ WP ]', '')
    
    prompts = []
    stories = []
    with open(source) as src, open(target) as tgt:
        prompts = src.readlines()
        stories = tgt.readlines()
    # select num_examples examples with [ WP ] tag and take out the tag!
    filtered = [(prompt, story) for prompt, story in zip(prompts, stories) if prompt.startswith('[ WP ]')]
    filtered = [filtered[idx] for idx in random.sample(range(len(filtered)), num_examples)]
    prompts, stories = zip(*filtered)
    prompts = [remove_prompt_tag(process_spaces(prompt)).strip() for prompt in prompts]
    stories = [process_spaces(story).strip() for story in stories]
    df = pd.DataFrame({'prompts': prompts, 'stories': stories})
    if outfile:
        df.to_csv(outfile, index=False)
    return df


def wp_generate(wp: pd.DataFrame, temp: float, prompt_msg='', min_words=200, outfile=None):
    """
    DESC: Another ChatGPT-generating function. No tokenization necessary 
    because we use the whole prompt to generate data, but optional 
    prompt_msg can be passed in.
    PARAMS: 
    wp: DataFrame with 'prompts' and 'stories' columns of human prompts and stories
    temp: temperature for sampling ChatGPT with
    prompt_msg: message to append to beginning of each prompt
    min_words: minimum words desired from each prompt
    outfile: where to save generated examples
    """
    wp['prompts'] = wp.apply(lambda row: prompt_msg + row['prompts'], axis=1)
    wp_with_responses = prompt_from_dataframe(wp, temp, min_words)
    if outfile:
        wp[['prompts', 'responses']].to_csv(outfile, index=False)
    return wp_with_responses

if __name__ == '__main__':
    argparser = ArgumentParser(prog='ChatGPT Scraper', description='Generate tokens and responses from ChatGPT using unofficial API.')
    argparser.add_argument('dataset', help="Specify which dataset you want to generate ChatGPT examples for.", choices=['xsum', 'wp', 'squad'])
    argparser.add_argument('-q', '--query', action='store_true', help='specify if you actually want to ask ChatGPT for examples. Safeguard against excess token use!')

    input = argparser.add_argument_group()
    input.add_argument('-l', '--load', action='store_true', help='if you need to download your dataset from Hub/files, specify this option')
    input.add_argument('-i', '--infile', help='files where dataset needs to be loaded from!')
    
    output = argparser.add_argument_group()
    output.add_argument('--out_human', help='If --load is specified, this is where load will store the human language data.')
    output.add_argument('--out_chatgpt', action='store', help='Destination file to write prompts/responses from ChatGPT.')

    prompt_opts = argparser.add_argument_group()
    prompt_opts.add_argument('-m', '--msg', help='prompt before \'actual\' dataset prompt to give ChatGPT, if that might help ChatGPT give a better response')
    prompt_opts.add_argument('-k', '--tokens', help='Specify number of tokens when creating prompts for XSum dataset.', default=30, type=int)
    prompt_opts.add_argument('-n', '--num_examples', help='Number of examples to grab when loading a dataset.', type=int, default=500)
    prompt_opts.add_argument('-w','--min_words', help='min_words desired from a ChatGPT response', type=int, default=250)
    prompt_opts.add_argument('-t', '--temperature', help='temperature for sampling ChatGPT', type=float)
    
    args = argparser.parse_args()

    openai.api_key = os.getenv('OPENAI_API_KEY')

    if args.dataset == 'xsum':
        xsum = xsum_load(infile=args.infile, outfile=args.out_human, num_examples=args.num_examples)
        if args.query:
            xsum_with_responses = xsum_generate(xsum, temp=args.temperature, tokens=args.tokens, prompt_msg=args.msg, min_words=args.min_words, outfile=args.out_chatgpt)

    elif args.dataset == 'squad':
        squad = squad_load(infile=args.infile, outfile=args.out_human, num_examples=args.num_examples)
        if args.query:
            squad_with_responses = squad_generate(squad, temp=args.temperature, min_words=args.min_words, outfile=args.out_chatgpt)

    elif args.dataset == 'wp':
        wp = wp_load(infile=args.infile, num_examples=args.num_examples, outfile=args.out_human, load=args.load)
        if args.query:
            wp_with_responses = wp_generate(wp, temp=args.temperature, prompt_msg=args.msg, min_words=args.min_words, outfile=args.out_chatgpt)

    print(f'Used {USAGE} tokens in this run.')