from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
import time
import openai
from tqdm import trange
import random

def gen_ref(num_ref, ref_size):
    # create this functio for more complex reference generation later
    samples = torch.randn(num_ref, ref_size)
    return samples

class pairwise_data_tokenized(Dataset):
    def __init__(self, annotated, tokenizer, model_max_length, prompts):
        self.data = []
        self.tokenizer = tokenizer
        # open with load json

        for entry in annotated:
            
            if entry['input'] == '':
                pair_1 = prompts['prompt_noinputs'].format(instruction=entry['instruction']) + entry['output_1']
                pair_2 = prompts['prompt_noinputs'].format(instruction=entry['instruction']) + entry['output_2']
            else:
                pair_1 = prompts['prompt_inputs'].format(instruction=entry['instruction'], input=entry['input']) + entry['output_1']
                pair_2 = prompts['prompt_inputs'].format(instruction=entry['instruction'], input=entry['input']) + entry['output_2']

            p_1 = self.tokenizer(pair_1, return_tensors="pt", padding='max_length', truncation=True, max_length=model_max_length)
            p_2 = self.tokenizer(pair_2, return_tensors="pt", padding='max_length', truncation=True, max_length=model_max_length)

            p_1_tk = p_1['input_ids'].squeeze(0)
            p_1_att = p_1['attention_mask'].squeeze(0)
            p_2_tk = p_2['input_ids'].squeeze(0)
            p_2_att = p_2['attention_mask'].squeeze(0)

            label = entry['preference']
            self.data.append((p_1_tk, p_2_tk, p_1_att, p_2_att, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
class PairwiseDyadicAugmentedTokenizedData(Dataset):
    def __init__(self, annotated, tokenizer, model_max_length, prompts):
        self.data = []
        self.tokenizer = tokenizer

        for dyadic_batch in annotated:
            p_1_tk_list = []
            p_1_att_list = []
            p_2_tk_list = []
            p_2_att_list = []
            label_list = []
            for _, dyadic_data in enumerate(dyadic_batch):

                if dyadic_data['input'] == '':
                    pair_1 = prompts['prompt_noinputs'].format(instruction=dyadic_data['instruction']) + dyadic_data['output_1']
                    pair_2 = prompts['prompt_noinputs'].format(instruction=dyadic_data['instruction']) + dyadic_data['output_2']
                else:
                    pair_1 = prompts['prompt_inputs'].format(instruction=dyadic_data['instruction'], input=dyadic_data['input']) + dyadic_data['output_1']
                    pair_2 = prompts['prompt_inputs'].format(instruction=dyadic_data['instruction'], input=dyadic_data['input']) + dyadic_data['output_2']

                p_1 = self.tokenizer(pair_1, return_tensors="pt", padding='max_length', truncation=True, max_length=model_max_length)
                p_2 = self.tokenizer(pair_2, return_tensors="pt", padding='max_length', truncation=True, max_length=model_max_length)


                p_1_tk = p_1['input_ids'].squeeze(0)
                p_1_att = p_1['attention_mask'].squeeze(0)
                p_2_tk = p_2['input_ids'].squeeze(0)
                p_2_att = p_2['attention_mask'].squeeze(0)
                label = dyadic_data['preference']
                p_1_tk_list.append(p_1_tk)
                p_1_att_list.append(p_1_att)
                p_2_tk_list.append(p_2_tk)
                p_2_att_list.append(p_2_att)
                label_list.append(torch.tensor(label))

            p_1_tk_batch = torch.stack(p_1_tk_list)
            p_1_att_batch = torch.stack(p_1_att_list)
            p_2_tk_batch = torch.stack(p_2_tk_list)
            p_2_att_batch = torch.stack(p_2_att_list)
            label_batch = torch.stack(label_list)     

            self.data.append((p_1_tk_batch, p_2_tk_batch, p_1_att_batch, p_2_att_batch, label_batch))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class single_data_tokenized(Dataset):
    def __init__(self, annotated, tokenizer, model_max_length, prompts):
        self.data = []
        self.tokenizer = tokenizer

        for entry in annotated:      

            pair_1 = prompts['prompt_inputs'].format(input=entry['input'])
            p_1 = self.tokenizer(pair_1, return_tensors="pt", padding='max_length', truncation=True, max_length=model_max_length)
            p_1_tk = p_1['input_ids'].squeeze(0)
            p_1_att = p_1['attention_mask'].squeeze(0)

            if entry['output'] == 'negative':
                label = torch.tensor([0.0])  
            elif entry['output'] == 'positive':
                label = torch.tensor([1.0])  

            self.data.append((p_1_tk, p_1_att, label))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class SingleDyadicAugmentedTokenizedData(Dataset):
    def __init__(self, annotated, tokenizer, model_max_length, prompts):
        self.data = []
        self.tokenizer = tokenizer

        for dyadic_batch in annotated:
            p_1_tk_list = []
            p_1_att_list = []
            label_list = []
            for _, dyadic_data in enumerate(dyadic_batch):

                pair_1 = prompts['prompt_inputs'].format(input=dyadic_data['input']) 

                p_1 = self.tokenizer(pair_1, return_tensors="pt", padding='max_length', truncation=True, max_length=model_max_length)

                p_1_tk = p_1['input_ids'].squeeze(0)
                p_1_att = p_1['attention_mask'].squeeze(0)

                if dyadic_data['output'] == 'negative':
                    label = 0  
                elif dyadic_data['output'] == 'positive':
                    label = 1 

                p_1_tk_list.append(p_1_tk)
                p_1_att_list.append(p_1_att)
                label_list.append(torch.tensor(label))

            p_1_tk_batch = torch.stack(p_1_tk_list)
            p_1_att_batch = torch.stack(p_1_att_list)
            label_batch = torch.stack(label_list)     

            self.data.append((p_1_tk_batch, p_1_att_batch, label_batch))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]