import pandas as pd
import openai
from tqdm import tqdm
import argparse
import os
from openai.error import RateLimitError
import time
from utils.prompter import Prompter

prompter = Prompter("gpt")
global tokens
tokens = 0

def make_query(row, args):
    global tokens

    content = prompter.generate_prompt(
        instruction=row['instruction'],
        topic=row['topic'], 
        objectives=row['objectives'],
        selection_criteria=row['selection_criteria'], 
        title=row['title'],
        abstract=row['abstract'])

    response = ask_model(content, args)
    gpt_response = response['choices'][0]['message']['content']
    tokens += response['usage']['total_tokens']

    return gpt_response

def ask_model(content, args):

    messages = {
        'role': 'assistant',
        'content': content
    }

    response = None
    attempts = 0
    while response is None:
        if attempts > 10:
            raise Exception("Too many failed attempts.")
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[messages],
                temperature=0,
                max_tokens=args.max_tokens,
                n=1
            )
        
        except RateLimitError as e:
            print("sleep", e)
            attempts += 1
            time.sleep(60)
        
        except Exception as e:  # http code 502 (bad gateway)
            print(e)
            time.sleep(10)
            attempts += 1

    return response

def main(args):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise Exception("OPENAI_API_KEY environment variable not set.")
    
    # Load data
    df = pd.read_json(args.data_path, orient='records')
    
    if 'gpt_response' not in df.columns:
        df['gpt_response'] = None
    
    tqdm.pandas()
    gpt_responses = df[df['gpt_response'].isna()].progress_apply(
        lambda x: make_query(x, args), axis=1)
    
    gpt_responses.name = 'gpt_response'
    df.update(gpt_responses)

    print(f'{tokens} tokens used.')
    
    print(f'{df["gpt_response"].isna().sum()} failed requests.')

    if not args.data_path.endswith('_gpt.json'):
        args.data_path = args.data_path.replace('.json', '_gpt.json')
    df.to_json(args.data_path, orient='records', indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data.json', help='Path to data json file')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens to use for GPT completion')
    args = parser.parse_args()
    main(args)