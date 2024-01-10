import numpy as np
import requests, json, os, time, random
from secret_things import openai_key

def readfile(path):
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json'):
            content = json.load(f)
        else:
            content = f.read()
    return content
def writefile(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, indent=2)
        else:
            f.write(content)

def embedder_api(strings):
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": strings,
        "model": "text-embedding-ada-002"
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)

    if response.status_code != 200:
        print(vars(response))
        raise Exception
    else:
        print(f'successfully embedded {len(strings)} strings')
    data = response.json()['data']
    return [d['embedding'] for d in data]

def save():
    writefile(path, existing_db)
def load():
    return readfile(path) if os.path.exists(path) else {}
def cleardb():
    writefile(path, {})
path = 'embeddings.json'
existing_db = load()

def update_database(strings):
    to_embed = [s for s in strings if s not in existing_db.keys()]
    print(f'will embed {len(to_embed)}/{len(strings)} strings')

    if to_embed != []:
        per_call = 50
        for i in range(0, len(to_embed), per_call):
            vectors = embedder_api(to_embed[i:i+per_call])
            for n, v in enumerate(vectors):
                idx = i*per_call + n
                string = to_embed[idx]

                existing_db[string] = v
        save()

def search(query):
    query_emb = embedder_api([query])
    strings = list(existing_db.keys())
    vectors = list(existing_db.values())

    triplets = sorted(
        [(
            n,
            strings[n],
            float(np.dot(query_emb,vectors[n])[0])
        ) for n in range(len(strings))],
        key=lambda triplet: triplet[2],
        reverse=True
    )
    return triplets

update_database([
    'banana',
    'kiwi',
    'look at me im a random sentence',
    'apple',
    'funny',
    'test',
])
while True:
    i = input('> ')
    print(search(i))
