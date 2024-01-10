import os
import json
import random
from tqdm import tqdm

import torch
import transformers
import logging
from sklearn.model_selection import StratifiedKFold

import openai
import numpy as np
from arguments import arg_parse
from train_predetector import train_detector
from train_disambiguator import train_disambiguator
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args = arg_parse()
edited_num = args.edited_num
dataset_name = args.dataset_name

retrain_detector = args.retraining_detector
cls_name = args.cls_name

retrain_disambiguator = args.retraining_disambiguator
seq_name = args.seq_name

use_kgprompt = args.activate_kgprompt

logging.basicConfig(filename=f'PokeMQA_result.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(f'PokeMQA on GPT-3.5-turbo-instruct')


def call_gpt(cur_prompt, stop):
  
    ans = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt= cur_prompt,
        temperature=0,
        stop=stop,
        max_tokens = 200
    )
    returned = ans['choices'][0]['text']
    return returned

def verify_subquestion_path(prompt,hops,path):
    # Checking the correctness of reasoning path i.e. Calculating Hop-Acc
    # subquestion verification
    if not len(prompt.strip().split('Subquestion:')) == hops+1:
        return 1
    
    # reasoning path verification
    sub_prompt = prompt.strip().split('Subquestion:')
    for idx in range(1,len(sub_prompt)):
        
        inter_ans = sub_prompt[idx].strip().split(': ')[-1]
        print(inter_ans)
        if inter_ans != path[idx-1]["answer"] and inter_ans not in path[idx-1]["answer_alias"]:
            return 2

    return False


# train or load scope detector
model_name = "distilbert-base-cased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

if retrain_detector:
    cls_name = train_detector()       # negative_sample_num = 20

if retrain_disambiguator:
    seq_name = train_disambiguator()    # negative_sample_num = 1

torch.cuda.empty_cache()

classifier = transformers.AutoModel.from_pretrained(f"detector-checkpoint/{cls_name}").to(device)
seq_cls = transformers.AutoModelForSequenceClassification.from_pretrained(f"detector-checkpoint/{seq_name}").to(device)

# loading dataset
with open(f'datasets/{dataset_name}.json', 'r') as f:
    dataset = json.load(f)      


# construct edit batches of different sizes
    
samples = 3000 if dataset_name=='MQuAKE-CF-3k' else 1868
dataset_splits = samples // edited_num  

hop_labels = [i//1000 for i in range(3000)]
if edited_num == 1 or edited_num == samples:
    hop_labels = [0 for i in range(samples)]

if edited_num == 1:
    skf = StratifiedKFold(n_splits= dataset_splits, shuffle=False)
elif edited_num != samples:
    skf = StratifiedKFold(n_splits= dataset_splits, shuffle=True, random_state=42)

subsets = []

if edited_num == samples:
    subsets.append([i for i in range(samples)])
else:
    for _, test_index in skf.split(dataset, hop_labels):
        subsets.append(test_index)

dataset_batch = []
edits_batch = []
embs_batch = []

for batch in subsets:

    sub_dataset = [dataset[index] for index in batch]
    new_facts = set()
    for d in sub_dataset:
        for r in d["requested_rewrite"]:
            if f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}' not in new_facts:
                new_facts.add(f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}')

    new_facts = list(new_facts)

    with torch.no_grad():
        facts_input = tokenizer(new_facts, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        facts_emb = classifier(**facts_input).last_hidden_state[:,0]

    dataset_batch.append(sub_dataset)
    embs_batch.append(facts_emb)
    edits_batch.append(new_facts)


# load inference prompts

with open('prompts/KGprompt.txt','r') as f:
    task_wkg_prompt = f.read()

with open('prompts/woKGprompt.txt','r') as f:
    task_wokg_prompt = f.read()


# load pregenerated knowledge prompt
if use_kgprompt:
    with open(f'kgprompt/{dataset_name}-kgprompt.json','r') as f:
        pre_kgprompt = json.loads(f.read())

# stop token
stop = ["Generated answer:","\n\n"]

# Inference stage
print(f"PokeMQA inference start (edit batch : {edited_num} instance):")
logging.info(f"PokeMQA inference start (edit batch : {edited_num} instance):")

T = 3000

cor = 0
ver_cor = 0
cor_list = [0,0,0]
ver_cor_list = [0,0,0]

tot = 0

for no in range(dataset_splits):
    for d in tqdm(dataset_batch[no]):
        tot += 1
        have_cor = False
        if use_kgprompt:
            kg_existing = pre_kgprompt[d["case_id"]-1]["existing"]
            kg_mapping = pre_kgprompt[d["case_id"]-1]["mapping"]

        for q in d["questions"]:
            found_ans = False
            prompt = task_wokg_prompt + "\n\nQuestion: "+ q
 
            if use_kgprompt and kg_existing[q]:
                prompt = task_wkg_prompt + "\n\nQuestion: "+ q + "\n" + "Entity of Question: " + kg_mapping[q]

            for i in range(5):
                # prompt the model to identify the subquestion
                gen = call_gpt(prompt, stop)

                last_sent = gen.strip().split('\n')[-1]
                # if final answer is there, get the answer and exit
                if last_sent.startswith('Final answer: '):
                    found_ans = True
                    ans = last_sent[len("Final answer: "):]
                    prompt = prompt + gen
                    break
                # otherwise, extract the generated subquestion
                if len(gen.strip().split('\n')) < 1 or len(gen.strip().split('\n')) > 3:
                    prompt = prompt + gen
                    break # failed case
                subquestion = gen.strip().split('\n')[-1]
                if not subquestion.startswith('Subquestion: '):
                    prompt = prompt + gen
                    break # failed case
                subquestion = subquestion[len("Subquestion: "):]
                
                # conflict detection
                with torch.no_grad():
                    subquestion_input = tokenizer(subquestion, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
                    subquestion_emb = classifier(**subquestion_input).last_hidden_state[:,0]
                    
                log_prob = (embs_batch[no]-subquestion_emb).norm(2,-1)
                log_prob = -log_prob**2

                prob = log_prob.exp()
    
                if prob.max() < 0.5:
                    # prompt = prompt + gen +'Intermediate answer:'
                    prompt = prompt + gen + "Generated answer"

                else:
                    idxs = prob >= 0.5
                    edits_mini = [edits_batch[no][i] for i in range(len(idxs)) if idxs[i]==True]

                    if len(edits_mini)>1:
                        input_batch = []
                        for can_edit in edits_mini:
                            input_batch.append(can_edit + tokenizer.sep_token + subquestion)
                        
                        with torch.no_grad():
                            batch_input = tokenizer(input_batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                            batch_logits  = seq_cls(**batch_input).logits
                            seq_prob = torch.softmax(batch_logits,-1)[:,0]
                        
                        if seq_prob.max() < 0.5:
                            prompt = prompt + gen + 'Generated answer'
                        
                        else:
                            value ,index = seq_prob.max(0)

                            # prompt = prompt + gen[:-remove_length] + edip ts_batch[no][index] + '.\nIntermediate answer:'
                            prompt = prompt + gen + 'Generated answer: '+edits_mini[index] + '.'

                    else:
                        value , index = prob.max(0)
                        prompt = prompt + gen + 'Generated answer: '+edits_batch[no][index] + '.'

            if not found_ans:
                continue
            prompt = prompt.strip().split('\n\n')[-1]
            
            # if the answer is correct -> positive instance for Acc
            if ans == d["new_answer"] or ans in d["new_answer_alias"]:   
                instance_type = verify_subquestion_path(prompt,len(d["single_hops"]),d["new_single_hops"])

                print('id:{} ans:{}'.format(d['case_id'],ans))
                logging.info('id:{} ans:{} progress:{}/{}/{}'.format(d['case_id'],ans,cor,cor_list[len(d["single_hops"])-2],tot))

                if not have_cor:
                    cor += 1
                    cor_list[len(d["single_hops"])-2] += 1
                    have_cor = True

                if not instance_type:         #  positive instance for Hop-Acc
                    ver_cor += 1
                    ver_cor_list[len(d["single_hops"])-2] += 1
                    print('verification passed sum:{} hop:{}'.format(ver_cor,ver_cor_list[len(d["single_hops"])-2]))
                    logging.info('passVerification sum:{} hop:{}'.format(ver_cor,ver_cor_list[len(d["single_hops"])-2]))
                    break


print(f'Acc = {cor / tot} ({cor} / {tot})')
print(f'Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})')

logging.info(f'Multi-hop Acc = {cor / tot} ({cor} / {tot})')
logging.info(f'2-hop = {cor_list[0]}')
logging.info(f'3-hop = {cor_list[1]}')
logging.info(f'4-hop = {cor_list[2]}')

logging.info(f'Hop-wise Acc (Hop-Acc) = {ver_cor / tot} ({ver_cor} / {tot})')
logging.info(f'2-hop = {ver_cor_list[0]}')
logging.info(f'3-hop = {ver_cor_list[1]}')
logging.info(f'4-hop = {ver_cor_list[2]}')