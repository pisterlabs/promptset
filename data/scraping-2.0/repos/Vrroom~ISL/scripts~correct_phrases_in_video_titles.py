import openai
import time
import json
import argparse
import pandas as pd
import os
from more_itertools import chunked, flatten
from isl_utils import *
import os
from tqdm import tqdm

openai.api_key = os.environ.get('OPENAI_KEY')

GPT_MODEL = "gpt-4"

prompt_begin = """
Here are some phrases from a dictionary on indian sign language. They have a bunch of mistakes. For example, 
some of them might have typos (e.g. `load` spelt as `laod`). Others may have random integers appended to them (e.g. 
`Basic Pay 1` instead of `Basic Pay`). Sometimes, there will be multiple phrases that mean the same thing but are
comma or space separated or something else (e.g. `hydrochloric acid HCl`, `Fir(First Information Report)` or 
`old, aged, elderly (person)`). There may be other mistakes which I don't know about.

I want you to fix these phrases. Give me the fixed list of phrases as a JSON. When the phrase has just one entity, 
return a singleton list for that entity. Otherwise, give a list. For example, `hydrochloric acid HCL` should become
[`hydrochloric acid`, `HCL`]. I'm giving you some examples below, following by the list:

Examples:

Letter Of Credit (Lc) -> [\"Letter of Credit\"]
scratch (skin) -> [\"scratch\"]
Pules Crops -> [\"Pulse Crops\"]
hair dryer (sign 1) -> [\"hair dryer\"]
in, under -> [\"in\", \"under\"]
Stimulator, Electric -> [\"Electric Stimulator\"]
crowd, horde, mass, throng -> [\"crowd\", \"horde\", \"mass\", \"throng\"]

Phrase List:

""" 
prompt_end = """

Please give me just the JSON (as a list of lists) and nothing else.
"""

def prepare_prompt (text_list) :
    return prompt_begin + '\n'.join(text_list) + prompt_end

@skip_if_processed()
def process_chunk(data) : 
    i, chunk = data
    while True: 
        try :
            prompt = prepare_prompt(chunk)
            response = openai.ChatCompletion.create( 
                model=GPT_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You cleanup text'},
                    {'role': 'user', 'content': prompt},
                ],
            )
            resp = response['choices'][0]['message']['content']
            processed_text_dict = dict(zip(chunk, eval(resp)))
            with open(str(i) + '_' + args.out_json, 'w+') as fp :
                json.dump(processed_text_dict, fp)
            break
        except Exception :
            continue

def main (args) : 
    unnormalized_text = pd.read_csv(args.unnormalized_text_file)
    text = f7(unnormalized_text['text'])
    chunks = list(chunked(text, 50))
    for i, chunk in enumerate(tqdm(chunks)) :
        process_chunk((i, chunk))

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Correct the text, fix typos etc")
    parser.add_argument('--out_json', required=True, type=str, help="Path to the video json directory.")
    parser.add_argument('--unnormalized_text_file', required=True, type=str, help="Path to the titles that may contain mistakes")
    args = parser.parse_args()
    main(args)

