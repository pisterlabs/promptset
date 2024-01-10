import concurrent.futures
import pandas as pd
import openai
import backoff
import os
import glob2
import numpy as np
import yaml

path = os.getcwd() #/home/suryabalaji/GPT_MolBERTa

with open(os.path.join(location, 'config_finetune.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

filename = os.path.join(path, 'data_gen', 'text_files_' + config['dataframe'])
api_key_path = path + '/' + 'GPT_API_Key.txt'

p = filename + '/' + '**.txt'

def done (path):
    new_list = []
    list_done = glob2.glob(path)
    print(len(list_done))
    for i in range(len(list_done)):
        name = list_done[i].split("/")[-1]
        mol_number = int(name.replace("Molecule ","").split(".")[0])
        new_list.append(mol_number)
    return new_list

path_to_dataset = os.path.join(location, 'datasets')
dataset_path = path_to_dataset + '/' + str(config['dataframe']) + '.csv'
length = len(pd.read_csv(dataset_path)['smiles'])

array = np.arange(1, length + 1).tolist()
done_p = done(p)
left_over = [i for i in array if i not in done_p]

openai.organization =      # 'org-SDGWEGKOMO352'
openai.api_key_path = api_key_path

MODEL = 'gpt-3.5-turbo'

def generate_completion(name, value):
    response = completions_with_backoff(
        model = MODEL,
        messages = [
            {"role": "system", "content": 'You are able to generate important and verifiable features about molecular SMILES'},
            {"role": "user", "content": f"Generate a description about the following SMILES molecule {value}"},
        ],
        temperature = 0,
        max_tokens = 2048
    )
    with open(filename + "/" + str(name) + '.txt', 'w') as f:
        f.write(str(response))
        
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)    
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def data_gen(left_over):
    df = pd.read_csv(path_to_dataset)
    x_data = df['smiles']
    mapping = {}    
    for idx in left_over:
        number = 'Molecule' + ' ' + str(idx)
        smiles = x_data[idx-1]
        mapping[number] = smiles

    max_threads = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers = max_threads) as executor:
        futures = [executor.submit(generate_completion, name, value) for name, value in mapping.items()]
        concurrent.futures.wait(futures)

data_gen(left_over)