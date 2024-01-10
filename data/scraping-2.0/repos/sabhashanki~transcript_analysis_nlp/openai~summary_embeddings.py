import os
import json
import openai
import pickle

openai.api_key = "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"


def get_embedding(text, model="text-similarity-davinci-001"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


path_to_json = '../data/output/summary/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        text = json_text['summary']
        print(f'{js} - loading')
        output = get_embedding(text)
        with open(f'./embeddings/summary_embeddings/{js}.pkl', 'wb') as file:
            pickle.dump(output, file)
            # print(f'{js} - Success')

