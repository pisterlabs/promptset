import pandas as pd
import pickle
from tqdm import tqdm
import time
from openai import OpenAI
import argparse
import requests
import heapq
from mFLAG.model import MultiFigurativeGeneration
from mFLAG.tokenization_mflag import MFlagTokenizerFast
import torch

parser = argparse.ArgumentParser(description='Argument Parser for Generating Using the Riddler Model')
parser.add_argument('--word', type=str, required=True, help='Enter a commonplace word or ')
parser.add_argument('--literal', action='store_true', help='Option for <literal>')
parser.add_argument('--hyperbole', action='store_true', help='Option for <hyperbole>')
parser.add_argument('--idiom', action='store_true', help='Option for <idiom>')
parser.add_argument('--sarcasm', action='store_true', help='Option for <sarcasm>')
parser.add_argument('--metaphor', action='store_true', help='Option for <metaphor>')
parser.add_argument('--simile', action='store_true', help='Option for <simile>')
parser.set_defaults(literal=True)
args = parser.parse_args()


device = f"cpu" if torch.cuda.is_available() else "cpu"

def get_related_concepts(word, limit=10):
    base_url = 'http://api.conceptnet.io/'
    search_url = f'{base_url}c/en/{word}?limit={limit}'
    related_concepts = []
    w_concepts = []

    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()

        edges = data['edges']
        for edge in edges:
            related_concepts.append(edge['end']['label'])
            w_concepts.append(edge['weight'])
    return related_concepts, w_concepts


def priority(starting_word, limit=10):
    priority_queue = []
    visited = set()
    concepts = []
    heapq.heappush(priority_queue, (0, starting_word))
    ct = 0
    while priority_queue:
        current_w, current_word = heapq.heappop(priority_queue)
        ct += 1
        if ct > limit:
            break

        if current_word not in visited:
            visited.add(current_word)
            concepts.append(current_word)

            related,w = get_related_concepts(current_word, limit)
            for related_word,wt in zip(related,w):
                heapq.heappush(priority_queue, (w, related_word))
    return concepts





client = OpenAI()



####### Concept Extraction
word = args.word
cp = priority(word, 20)[1:]


####### Base Riddle Generation
generation = ""
if len(cp)!=0:
    msg = f"Can you create a short riddle for the word: {word} which touches upon these concepts: {cp[:4]} ?"
else:
    msg = f"Can you create a short riddle for the word: {word} ?"

completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      messages=[
            {"role": "user", "content": msg}
      ]
)

generation = str(completion.choices[0].message.content)


## Embellishment
tokenizer = MFlagTokenizerFast.from_pretrained('laihuiyuan/mFLAG')
model = MultiFigurativeGeneration.from_pretrained('laihuiyuan/mFLAG')
model = model.to(device)


# spcl_token = "<literal>"
# if args.hyperbole:
#     spcl_token = "<hyperbole>"
# elif args.idiom:
#     spcl_token = "<idiom>"
# elif args.sarcasm:
#     spcl_token = "<sarcasm>"
# elif args.metaphor:
#     spcl_token = "<metaphor>"
# elif args.simile:
#     spcl_token = "<simile>"

final_riddles = []
embs = ["<hyperbole>","<idiom>","<sarcasm>","<metaphor>","<simile>"]
for spcl_token in embs:
      inp_ids = tokenizer.encode(generation, return_tensors="pt")
      # the target figurative form (<sarcasm>)
      fig_ids = tokenizer.encode(spcl_token, add_special_tokens=False, return_tensors="pt")
      inp_ids = inp_ids.to(device)
      fig_ids = fig_ids.to(device)
      outs = model.generate(input_ids=inp_ids[:, 1:], fig_ids=fig_ids, forced_bos_token_id=fig_ids.item(), num_beams=10, max_length=512)
      final_riddle = tokenizer.decode(outs[0, 2:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
      final_riddles.append(final_riddle)

print("Concepts Extracted: ")
for i,c in enumerate(cp):
    print(str(i)+'.'+str(c))
print(10*"<<"+"Base Riddle:"+10*">>")
print(generation)
for emb, final_riddle in zip(embs, final_riddles):
      print(10*"<<"+f"Final Riddle: (w {emb})"+10*">>")
      print(final_riddle)
