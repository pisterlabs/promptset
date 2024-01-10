import gzip
import os
import sys
import pickle
import json
import time
import glob
import random
import shutil
import pathlib
import logging
import datetime
import argparse
import pandas as pd
import openai
from distutils.util import strtobool



MODEL_MAPPING = {
    'instructgpt': 'text-davinci-003',
    'gpt3': 'text-davinci-002',
    'text-davinci-003': 'text-davinci-003',
    'text-davinci-002': 'text-davinci-002',
    'code-davinci-002': 'code-davinci-002',
    'chatGPT' : 'gpt-3.5-turbo'
}

# Configs
logger = logging.getLogger('logger')

def load_parser_and_args():
    parser = argparse.ArgumentParser()
    ### directory ###
    parser.add_argument('--base_dir', type=str, default='/home/intern/sblee/sblee/Samsung')
    parser.add_argument('--task_dir', type=str, default='/home/intern/sblee/sblee/Samsung/rationale/data/preprocessed_te_apoe_q21.pickle')
    parser.add_argument('--prompt', type=str, default='prompt6')

    ### model parameters ###
    parser.add_argument('--model_type', type=str, default='chatGPT')
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--num_samples', type=int, default=0)

    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.base_dir, 'results')
    args.model_name_or_path = MODEL_MAPPING[args.model_type]
    return parser, args



def init_logger(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    handler = logging.FileHandler(os.path.join(args.output_dir, '{}_{:%Y-%m-%d-%H:%M:%S}.log'.format(args.prompt, datetime.datetime.now())), encoding='utf=8')
    logger.addHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.warning(args)



def corr_ans(inp=str):
    inp = inp.lower().replace('\n','')
    output = list(inp.split())
    output_len = len(output)
    frt_output = " ".join(output[:])
    answer = 0
    
    if 'normal cognition' in frt_output : answer = 'CN'
    elif 'mild cognitive impairment' in frt_output : answer = 'MCI'
    elif 'dementia' in frt_output : answer = 'Dementia'
    else : 
        if output_len < 7 :
            if 'normal' in inp : answer = 'CN'
            elif 'mild' in inp : answer = 'MCI'
            elif 'dementia' in inp : answer = 'Dementia'
        else : 
            if 'normal' in frt_output : answer = 'CN'
            elif 'mild' in frt_output : answer = 'MCI'
            elif 'dementia' in frt_output : answer = 'Dementia'
        
    return answer



def Accuracy(prediction):

    filenames = [k for k, v in prediction.items()]
    groundtruths = [v['groundtruth'] for k, v in prediction.items()]
    predictions = [corr_ans(v['prediction']) for k, v in prediction.items()]

    df = pd.DataFrame(zip(filenames, groundtruths, predictions), columns=['file name', 'groundtruth', 'prediction'])
    df['accurate'] = df['prediction'] == df['groundtruth']

    return df['accurate'].sum()/len(df)



class GPT(object):
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.frequency_penalty = args.frequency_penalty
        self.presence_penalty = args.presence_penalty
        self.cur_idx = -1
        self.cur_req = 0

    def login_to_openai(self, keys, cur_idx):
        openai.api_key = keys[cur_idx] 

    def set_new_key(self):
        with open('keys.json') as f:
            keys = json.load(f)
        self.cur_idx += 1
        self.cur_idx = self.cur_idx % len(keys)
        self.login_to_openai(keys, self.cur_idx)

    def inference(self, prompt, return_raw=False):
        timeout_stack = 0
        while True:
            if self.cur_req >= 15:
                time.sleep(60)
                self.cur_req = 0
            try:
                if self.model_name == 'gpt-3.5-turbo':                  # chatGPT
                    output = openai.ChatCompletion.create(               
                        model = self.model_name,
                        messages=[
                            {"role":"system", "content": prompt},
                        ]
                    )
                    break
                else :
                    output = openai.Completion.create(                     
                        engine=self.model_name,
                        prompt=prompt,
                        n=1, # How many completions to generate for each prompt.
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        logprobs=1
                    )
                    break

            except Exception as e:
                timeout_stack += 1
                if timeout_stack >= 3:
                    logger.info("Change to another key")
                    self.set_new_key()
                    timeout_stack = 0
                time.sleep(60)
        if return_raw:
            return output

        if self.model_name == 'gpt-3.5-turbo' :
            return output['choices'][0]['message']['content']           # chatGPT
        else : 
            return output['choices'][0]['text']                        



def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
 
    # setup logging
    init_logger(args)

    # load model
    prediction = dict()
    model = GPT(args)
    model.set_new_key()

    # load data
    with gzip.open(args.task_dir, 'r') as f:
        fr = pickle.load(f)
    

    for t in fr:
        idx = t['file name']
        input_dict = {}
        name_list = ['label', 'age', 'sex', 'educ', 'marriage', 'apoe', 'mmse', 'fmri']
        for name in name_list:
            input_dict[name] = t[name]
        for id in range(22):
            input_dict['q{}'.format(id)] = t['q{}'.format(id)]

        # prompt making
        with open(os.path.join(args.base_dir, 'prompt', '{}.json'.format(args.prompt)), 'r') as f:
            prompt = f.read()
        model_input = prompt.format(**input_dict)

        logger.info("***** Model Input *****")
        logger.info(model_input)

        # inference
        pred = model.inference(model_input)
        logger.info("***** Model Output *****")
        logger.info({input_dict['label'] : pred})

        # saving
        prediction[idx] = {'groundtruth' : input_dict['label'] , 'prediction' : pred}
 
    # accuracy
    accr = Accuracy(prediction)
    logger.info("accuracy: {}".format(accr))

    result = {'accuracy' : accr, 'predictions' : prediction }

    with open(os.path.join(args.output_dir, '{}_{:%Y-%m-%d-%H:%M:%S}_predicted_results.json'.format(args.prompt, datetime.datetime.now())), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return accr

if __name__ == "__main__":
    parser, args = load_parser_and_args()
    main(args)