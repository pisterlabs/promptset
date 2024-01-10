import argparse
import os
import sys
from time import sleep
from typing import Any, List, Dict
import openai
from tqdm.auto import tqdm
from datasets import load_dataset

from db import DB
from ai import embedding_of_many, embedding_of, complete


openai.api_key = open('openai.key', 'r').read().splitlines()[0]
embed_model = "text-embedding-ada-002"
db = DB("demo", "youtube_transcriptions")

parser = argparse.ArgumentParser(description="Retrieve data and generate answers")
parser.add_argument('--load_data', action='store_true', help="Load data if required")
args = parser.parse_args()

if args.load_data:
    data = load_dataset('jamescalam/youtube-transcriptions', split='train')

    new_data = []
    window = 20  # number of sentences to combine
    stride = 4  # number of sentences to 'stride' over, used to create overlap

    print('loading data')
    for i in tqdm(range(0, len(data), stride)):
        i_end = min(len(data)-1, i+window)
        if data[i]['title'] != data[i_end]['title']:
            # in this case we skip this entry as we have start/end of two videos
            continue
        text = ' '.join(data[i:i_end]['text'])
        # create the new merged dataset
        new_data.append({
            'start': data[i]['start'],
            'end': data[i_end]['end'],
            'title': data[i]['title'],
            'text': text,
            'id': data[i]['id'],
            'url': data[i]['url'],
            'published': data[i]['published'],
            'channel_id': data[i]['channel_id']
        })
    print(f"Created {len(new_data)} new entries from {len(data)} original entries")

    batch_size = 100  # how many embeddings we create and insert at once

    print('generating embeddings')
    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end of batch
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
        embeds = embedding_of_many(texts, embed_model)
        db.upsert_batch(meta_batch, embeds)

# Now we search
query = f"Which training method should I use for sentence transformers when " + \
        f"I only have pairs of related sentences?"

print(f'Query? (enter to use default)\n\nDefault:\n{query}\n\n> ', end='')
query = sys.stdin.readline().strip() or query

def enrich(query):
    # get relevant contexts (including the questions) and add them to the openai prompt
    limit = 3750
    xq = embedding_of(query, embed_model)
    contexts = db.query(xq, top_k=3)
    print(f"Retrieved {len(contexts)} contexts after asking for 3")

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts) + 1):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt_middle = "\n\n---\n\n".join(contexts[:i-1])
            break
    else:
        prompt_middle = "\n\n---\n\n".join(contexts)

    return prompt_start + prompt_middle + prompt_end

# first we retrieve relevant items from the database
query_with_contexts = enrich(query)
# then we complete the context-infused query
print(query_with_contexts)
response = complete(query_with_contexts)
print(f'Answer: {response}')
