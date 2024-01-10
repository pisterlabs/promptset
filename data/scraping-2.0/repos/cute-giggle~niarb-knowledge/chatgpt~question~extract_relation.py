import openai
import os
import json
import time

from openai.error import RateLimitError

os.environ['HTTP_PROXY'] = '192.168.1.34:7890'
os.environ['HTTPS_PROXY'] = '192.168.1.34:7890'

openai.api_key_path = '../data/api-key.txt'

def call(content: str):
    message = [
        {'role': 'user', 'content': content},
    ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=message,
        temperature=0,
    )
    return response['choices'][0]['message']['content']

def read_corpus():
    with open('data/corpus.json', 'r') as f:
        return json.load(f)
    
def read_relations():
    if not os.path.exists('data/relations.json'):
        return {}
    with open('data/relations.json', 'r') as f:
        return json.load(f)
    
def save_relations(relations):
    with open('data/relations.json', 'w') as f:
        json.dump(relations, f)

corpus = read_corpus()
relations = read_relations()
region_names = list(corpus.keys())

no_names = []
for region_name in region_names:
    if region_name not in relations:
        no_names.append(region_name)

region_names = no_names

i = 0
while i < len(region_names):
    region_name = region_names[i]
    prompt = f"""
    Your task is to extract brain science field entities and relationships from text delimited with triple backticks as accurate as possible and organize them into relational triples.
    The text is as follows: '''{json.dumps(corpus[region_name])}'''.
    The relational triplet format is as follows: [["entity1", "relationship", "entity2"], ...]
    If you can't answer, you must only output []
    """

    try:
        response = call(prompt)
        relations[region_name] = json.loads(response)
    except RateLimitError:
        time.sleep(10)
        continue
    except:
        save_relations(relations)
        raise

    i += 1
    print(f'count: {i}, region_name: {region_name}, response: {response}')

save_relations(relations)
