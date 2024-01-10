import os
import openai
import json
from tqdm import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('modelGPT: encoding text prompts', add_help=False)
    parser.add_argument('--model_config_name', default='LOVM/models.yml', type=str)
    parser.add_argument('--datasets', default='LOVM/datasets.txt', type=str)
    parser.add_argument('--api_key', default='', type=str)
    parser.add_argument('--num_captions', default=50, type=int)
    parser.add_argument('--captions_dir', default='modelGPT/captions_dataset', type=str)
    return parser

    

def main(args):
    openai.api_key = args.api_key
    num_captions = args.num_captions
    os.makedirs(args.captions_dir, exist_ok=True)

    with open(args.datasets, 'r') as f:
        datasets = [line.rstrip('\n') for line in f.readlines()]
    
    with open('LOVM/dataset_domains.json', 'r') as f:
        dataset_domains = json.load(f)

    with open('LOVM/dataset_tasks.json', 'r') as f:
        dataset_tasks = json.load(f)

        text_prompt = lambda c, d, t, n: f"Generate long, diverse, and confusing image captions for the {d} domain, which will be used to evaluate {t}. \nTry different prefixes, e.g., 'A/An/photo/photograph/image/{d}/etc' and varying caption lengths. \nGenerate {n} long domain-specific captions for class '{c}':"

    

    for d in tqdm(datasets, desc="Dataset"):
        res={}
        with open(f'LOVM/classnames/{d}.txt', 'r') as file:
            lines = file.readlines()

        classnames = [line.rstrip() for line in lines]
        
        for c in tqdm(classnames, desc=d+', Class', leave=False):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages= [{"role": "user", "content":text_prompt(c, dataset_domains[d], dataset_tasks[d], num_captions)}],
                    temperature=1,
                    max_tokens=2789,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
            except:
                print("############## fail #1, pausing and trying again")
                import time
                time.sleep(150)
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages= [{"role": "user", "content":text_prompt(c, dataset_domains[d], dataset_tasks[d], num_captions)}],
                        temperature=1,
                        max_tokens=2789,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    ) 
                except:
                    print([{"role": "user", "content":text_prompt(c, dataset_domains[d], dataset_tasks[d], num_captions)}])
                
            res[c] = response['choices'][0]['message']['content'].split('\n')
            
        with open(f"{args.captions_dir}/{d}.json", "w") as f:
            json.dump(res, f)   
            
    for d in tqdm(datasets, desc="Dataset"):
        with open(f"{args.captions_dir}/{d}.json", "r") as f:
            res = json.load(f)
            
        for c, v in res.items():
            v= [vv for vv in v if len(vv)>5]
            res[c] = [vv.split(f'{i+1}. ')[1] for i, vv in enumerate(v)]
            
        with open(f"{args.captions_dir}/{d}.json", "w") as f:
            json.dump(res, f)   
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()    
    main(args)
