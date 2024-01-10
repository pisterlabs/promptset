import os
import openai
import json
from tqdm import tqdm
import argparse
import time


def get_args_parser():
    parser = argparse.ArgumentParser('modelGPT: encoding text prompts', add_help=False)
    parser.add_argument('--model_config_name', default='LOVM/models.yml', type=str)
    parser.add_argument('--datasets', default='LOVM/datasets.txt', type=str)
    parser.add_argument('--api_key', default='', type=str)
    parser.add_argument('--syn_dir', default='modelGPT/syn_dataset', type=str)
    return parser

    

def main(args):
    openai.api_key = args.api_key
    os.makedirs(args.syn_dir, exist_ok=True)

    with open(args.datasets, 'r') as f:
        datasets = [line.rstrip('\n') for line in f.readlines()]
    
    with open('LOVM/dataset_domains.json', 'r') as f:
        dataset_domains = json.load(f)

    with open('LOVM/dataset_tasks.json', 'r') as f:
        dataset_tasks = json.load(f)
       
    text_prompt = lambda c, d, t:f"""In {d} {t} and for the class: {c}
    what are the appropriate superclasses/synonyms, e.g., 
    chihuahua: 1. animal, 2. dog, 3. small dog.
    chair: 1. object, 2. furniture, 3. seat, 4. armchair, 5. stool, 6. ottoman, 7. rocking chair.  
    The appropriate superclasses/synonyms for {c}:""" 


    for d in tqdm(datasets, desc="Dataset"):
        if f'{d}.json' in os.listdir(args.syn_dir):
            continue
        res={}
        with open(f'LOVM/classnames/{d}.txt', 'r') as file:
            lines = file.readlines()

        classnames = [line.rstrip() for line in lines]
        
        for c in tqdm(classnames, desc=d+', Class', leave=False):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages= [{"role": "user", "content":text_prompt(c, dataset_domains[d], dataset_tasks[d])}],
                    temperature=0.1,
                    max_tokens=2789,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except:
                time.sleep(150)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages= [{"role": "user", "content":text_prompt(c, dataset_domains[d], dataset_tasks[d])}],
                    temperature=0.1,
                    max_tokens=2789,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
            res[c] = response['choices'][0]['message']['content'].split('\n')
            
        with open(f"{args.syn_dir}/{d}.json", "w") as f:
            json.dump(res, f)   
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()    
    main(args)
