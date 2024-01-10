import json

import openai
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

from multi_type_search.utils.paths import DATA_FOLDER
from multi_type_search.search.graph.graph import Graph
from jsonlines import jsonlines


# Set up OpenAI API key
openai.api_key = ""


def get_gpt3_embeddings(text_list):
    embeddings = []
    for text in tqdm(text_list, desc='Embedding with openai', total=len(text_list)):
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        embedding = np.array(response['data'][0]['embedding'])
        embeddings.append(embedding)
    return np.array(embeddings)


def save_embeddings_to_file(embeddings, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(embeddings, f)


def load_embeddings_from_file(file_name):
    with open(file_name, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

tree_files = [
    DATA_FOLDER / 'full/morals/moral100.json',

]
emb_file = 'moral_gpt3_embeddings.pkl'
str_file = 'moral_gpt3_strings.json'

graphs = []

for tree_file in tree_files:
    if str(tree_file).endswith('.jsonl'):
        data = list(jsonlines.open(str(tree_file), 'r'))
    else:
        data = json.load(tree_file.open('r'))

    graphs.extend([Graph.from_json(t) for t in data])

strings = [x.normalized_value for y in graphs for x in y.premises]
strings.extend([x.normalized_value for z in graphs for y in z.deductions for x in y.nodes])
strings = list(set(strings))

# Replace with your own list of strings
# text_list = ["Hello world", "I love programming", "OpenAI GPT-3 is amazing"]
text_list = strings



json.dump(strings, Path(f'./{str_file}').open('w'))

# Get the embeddings
# embeddings = get_gpt3_embeddings(text_list)

# Save embeddings to a file
# save_embeddings_to_file(embeddings, emb_file)

# Load embeddings from the file
loaded_embeddings = load_embeddings_from_file(emb_file)
print("Loaded embeddings:", loaded_embeddings)