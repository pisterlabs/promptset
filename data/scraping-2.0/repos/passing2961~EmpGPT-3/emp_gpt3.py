from typing import List
import os
import json
import openai
from regex import P
from tqdm import tqdm
import pickle as pc
import random
from copy import deepcopy
import argparse
from collections import defaultdict
import torch
import numpy as np

from sentence_transformers import SentenceTransformer, util

os.environ["TOKENIZERS_PARALLELISM"] = "true"


openai.organization = "<YOUR ORG>"
# Load your API key from an environment variable or secret management service
openai.api_key = "<YOUR KEY>"


class Prompt(object):

    def __init__(self, prompt_path: str, spk1: str, spk2: str, 
        do_fewshot: bool, k: int, fewshot_data: List, fewshot_type: str
    ):

        self.prompt_path = prompt_path
        with open(prompt_path, 'r') as f:
            self.init_prompt = f.read()
        
        self.spk1 = spk1
        self.spk2 = spk2

        self.do_fewshot = do_fewshot
        self.k = k
        self.fewshot_data = fewshot_data
        self.reverse_order = True
        self.fewshot_type = fewshot_type

        if self.do_fewshot:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SentenceTransformer('stsb-roberta-large').to(self.device)

            if self.fewshot_type == 'situation':
                # data {conv_idx: situation} format
                fewshot_data = []
                
                for conv in self.fewshot_data:
                    for example in conv:
                        situation = example['situation']
                    fewshot_data.append(situation)

                self.fewshot_embedding = self.model.encode(fewshot_data, convert_to_tensor=True, device=self.device)
            elif self.fewshot_type == 'emotion':
                fewshot_data = defaultdict(list)

                for conv in self.fewshot_data:
                    for example in conv:
                        conv_idx = example['conv_idx']
                        emotion = example['emotion']
                        situation = example['situation']
                    
                    fewshot_data[emotion].append((situation, conv_idx))

                fewshot_emb = {}
                for emo, _conv in fewshot_data.items():
                    conv = [ele[0] for ele in _conv]
                    fewshot_emb[emo] = self.model.encode(conv, convert_to_tensor=True, device=self.device)

                self.fewshot_embedding = fewshot_emb
                self.fewshot_data_for_emo = fewshot_data

    def _prepare_fewshot_embedding(self, fewshot_data):
        results = {}
        for conv_idx, situation in tqdm(fewshot_data.items(), total=len(fewshot_data)):
            sit_embedding = self.model.encode(situation, convert_to_tensor=True, device=self.device)
            results[conv_idx] = sit_embedding
        
        return results
    
    def get_test_embedding(self, test_situation):
        return self.model.encode(test_situation, convert_to_tensor=True, device=self.device)

    def get_cosine_sim(self, test_embedding, emotion=None):
        if emotion is not None:
            fewshot_embedding = self.fewshot_embedding[emotion]
        else:
            fewshot_embedding = self.fewshot_embedding

        cosine_scores = util.cos_sim(test_embedding, fewshot_embedding)[0].detach().cpu().numpy()
        sort = np.argsort(cosine_scores)[::-1]
        return sort[:self.k], [cosine_scores[idx] for idx in sort[:self.k]]
        
    def make_fewshot_prompt_input(self, test_situation, emotion=None):
        
        test_embedding = self.get_test_embedding(test_situation)
        fewshot_indices, fewshot_sim_scores = self.get_cosine_sim(test_embedding, emotion)

        fewshot_prompt = []
        fewshot_sim_results = []
        fewshot_check_results = []
        for fewshot_idx, fewshot_sim_score in zip(fewshot_indices, fewshot_sim_scores):
            input_prompt = deepcopy(self.init_prompt)

            if self.fewshot_type == 'situation':
                conv = self.fewshot_data[fewshot_idx]
            elif self.fewshot_type == 'emotion':
                _, conv_idx = self.fewshot_data_for_emo[emotion][fewshot_idx]
                conv = self.fewshot_data[conv_idx]
                
            for i, example in enumerate(conv):
                utter = example['utter']
                emotion = example['emotion']

                if i % 2 == 0:
                    spk = self.spk1
                else:
                    spk = self.spk2

                input_prompt += f'{spk}: {utter}\n'

            fewshot_prompt.append(input_prompt)
            fewshot_sim_results.append(fewshot_sim_score)
            fewshot_check_results.append([example['situation'], example['emotion']])

        if self.reverse_order:
            fewshot_prompt.reverse()
            fewshot_sim_results.reverse()
            fewshot_check_results.reverse()
        
        fewshot_prompt_text = '\n'.join(fewshot_prompt)
        
        return fewshot_prompt_text, fewshot_sim_results, fewshot_check_results

    def make_fewshot_prompt_input_random(self):
        sampled_fewshot_data = random.sample(self.fewshot_data, self.k)

        fewshot_prompt = []
        for conv in sampled_fewshot_data:
            input_prompt = deepcopy(self.init_prompt)

            for i, example in enumerate(conv):
                utter = example['utter']
                emotion = example['emotion']

                if i % 2 == 0:
                    spk = self.spk1
                else:
                    spk = self.spk2

                input_prompt += f'{spk}: {utter}\n'
            fewshot_prompt.append(input_prompt)
        fewshot_prompt_text = '\n'.join(fewshot_prompt)
        return fewshot_prompt_text

    def make_prompt_input(self, conv_data):

        results = []
        cnt = 0
        check_results = []
        for conv in tqdm(conv_data):
            cnt += 1
            
            if self.do_fewshot:
                if self.fewshot_type == 'situation':
                    situation = conv[0]['situation']
                    fewshot_prompt_input, fewshot_sim_results, fewshot_check_results = self.make_fewshot_prompt_input(situation)
                    
                    check_results.append({
                        'prompt_input': fewshot_prompt_input,
                        'target_sit': situation,
                        'target_emo': conv[0]['emotion'],
                        'ex': fewshot_check_results,
                        'sim': fewshot_sim_results,
                    })
                    
                    #assert 1 == 0
                elif self.fewshot_type == 'emotion':
                    emotion = conv[0]['emotion']
                    situation = conv[0]['situation']
                    fewshot_prompt_input, fewshot_sim_results, fewshot_check_results = self.make_fewshot_prompt_input(situation, emotion)

                    check_results.append({
                        'prompt_input': fewshot_prompt_input,
                        'target_sit': situation,
                        'target_emo': emotion,
                        'ex': fewshot_check_results,
                        'sim': fewshot_sim_results,
                    })
                elif self.fewshot_type == 'random':
                    fewshot_prompt_input = self.make_fewshot_prompt_input_random()
                input_prompt = fewshot_prompt_input + '\n' + deepcopy(self.init_prompt)
            else:
                input_prompt = deepcopy(self.init_prompt)
            
            for i, example in enumerate(conv):
                utter = example['utter']
                emotion = example['emotion']

                if i % 2 == 0:
                    spk = self.spk1
                else:
                    spk = self.spk2

                if i == len(conv) - 1:
                    input_prompt += f'{spk}:'
                    gold_resp = utter
                    gold_emo = emotion
                else:
                    input_prompt += f'{spk}: {utter}\n'
            _ret = {
                'prompt_input': input_prompt,
                'dialog': conv,
                'gold_resp': gold_resp,
                'gold_emo': gold_emo
            }
            if self.do_fewshot and self.fewshot_type != 'random':
                _ret['fewshot_sim_scores'] = fewshot_sim_results

            results.append(_ret)
        
        assert len(results) == len(conv_data)
        return results
    
    def make_openai_call(self, prompt_input, stop_seq: List[str]):

        temp = 0.8
        max_tokens = 128
        top_p = 1
        freq_penalty = 0.4
        pres_penalty = 0.4
        #stop_seq = ['###']
        
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt_input,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            stop=stop_seq,
        )
        
        resp = response['choices'][0]['text']

        return resp[1:].strip()

def make_conversation(data):
    conv_data, tmp = [], []
    prev_conv_idx = 0
    for i, example in enumerate(data):
        conv_idx = example['conv_idx']
        utter_idx = example['utter_idx']
        utter = example['utterance']
        emotion = example['emotion']
        situation = example['situation']
        
        if prev_conv_idx != conv_idx:
            conv_data.append(tmp)
            tmp = []
        
        prev_conv_idx = conv_idx
        tmp.append({'conv_idx': conv_idx, 'utter': utter, 'emotion': emotion, 'situation': situation})

    conv_data.append(tmp)
    return conv_data

def prepare_train_for_fewshot():
    data_dir = f'./data/annotated_data/empathetic_dialogues/train.jsonl'

    with open(data_dir, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]

    return make_conversation(data)

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--do-fewshot', action="store_true")
    parser.add_argument('--k', default=0, type=int)
    parser.add_argument('--do-empgpt3', action="store_true")
    parser.add_argument('--fewshot-type', default='situation')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    datatype = 'test'
    data_dir = f'./data/annotated_data/empathetic_dialogues/{datatype}.jsonl'

    with open(data_dir, 'r') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]

    # make conversation dataset
    conv_data = make_conversation(data)

    fewshot_data = prepare_train_for_fewshot()
    print(len(fewshot_data))

    args = _parse_args()
    
    if args.do_empgpt3:
        prompt_dir = './prompt/prompt.txt'
        spk1 = 'Human'
        spk2 = 'Empathy AI'
        prompt_typ = 'empathy'
    else:
        prompt_dir = './prompt/vanilla_prompt.txt'
        spk1 = 'Human'
        spk2 = 'AI'
        prompt_typ = 'vanilla'

    prompt = Prompt(prompt_dir, spk1, spk2, args.do_fewshot, args.k, fewshot_data, args.fewshot_type)
    results = prompt.make_prompt_input(conv_data)
    
    result_save_dir = f'./result/davinci/empgpt3_{prompt_typ}_prompt_fewshot_{args.k}_{args.fewshot_type}_all/{datatype}'
    os.makedirs(result_save_dir, exist_ok=True)

    ret_val = []
    for result in tqdm(results):
        resp = prompt.make_openai_call(result['prompt_input'], stop_seq=[f'{spk1}:', f'{spk2}:'])
        
        result.update({'pred_resp': resp})
        
        ret_val.append(result)
        
        
    with open(os.path.join(result_save_dir, 'results.pkl'), 'wb') as f:
        pc.dump(ret_val, f)