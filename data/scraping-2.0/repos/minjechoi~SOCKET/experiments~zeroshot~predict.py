import os
import sys
import argparse
import json
import math
import re
import string
import collections
from getpass import getpass

import torch
from langchain.llms import OpenAI
from transformers import (
    AutoConfig,
    pipeline
)
from langchain.chat_models import ChatOpenAI
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
from transformers.pipelines.pt_utils import KeyDataset
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

print('PID: ',os.getpid())

parser = argparse.ArgumentParser("")
parser.add_argument("--model_type", type=str, default='huggingface')
parser.add_argument("--model_name_or_path", type=str, default='google/flan-t5-small')
parser.add_argument("--data_name_or_path", type=str, default='Blablablab/SOCKET')
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--data_split", type=str, default='test')
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_len", default=2048, type=int)
parser.add_argument("--max_new_tokens", default=100, type=int)
parser.add_argument("--tasks", type=str, default='ALL')
parser.add_argument("--result_path", default='results/')
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--use_sockette", action="store_true")
parser.add_argument("--unit_test", action="store_true")
parser.add_argument("--debug", action="store_true")



# functions for normalizing texts
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

# functions for computing scores
def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        res = int(gold_toks == pred_toks)
        return res, res, res
    if num_same == 0:
        return 0,0,0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def get_mean(li):
    return sum(li)/len(li)

def get_all_f1(groundtruth, answer):
    f1 = get_mean([compute_f1(g,a)[0] for a,g in zip(answer,groundtruth) ])
    p = get_mean([compute_f1(g,a)[1] for a,g in zip(answer,groundtruth) ])
    r = get_mean([compute_f1(g,a)[2] for a,g in zip(answer,groundtruth) ])
    return p, r, f1

def data_iterator(data, batch_size = 64):
    n_batches = math.ceil(len(data) / batch_size)
    for idx in range(n_batches):
        x = data[idx *batch_size:(idx+1) * batch_size]
        yield x

# def get_max_seq_len(model_id):
#     model_id = model_id.lower()
#     if 'opt-' in model_id:
#         return 2048
#     elif 'bloom' in model_id:
#         return 2048
#     elif 'gpt' in model_id:
#         return 2048
    
#     else:
#         return 2048

def truncate(sen, tokenizer, max_length=512):
    en_sen = tokenizer.encode(sen)
    sen = tokenizer.decode(en_sen[:max_length])
    return sen

args = parser.parse_args()
print(args)

# modify transformers cache
if args.model_cache_dir:
    os.environ['TRANSFORMERS_CACHE'] = args.model_cache_dir


if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

# load prompts
ppt_df = pd.read_csv('socket_prompts.csv')

if args.tasks in ['CLS','REG','PAIR','SPAN']:
    tasks_df = ppt_df[ppt_df['type']==args.tasks]
elif args.tasks == 'ALL':
    tasks_df = ppt_df
elif args.tasks in set(ppt_df['task']):
    tasks_df = ppt_df[ppt_df['task']==args.tasks]
elif ',' in args.tasks:
    tasks = args.tasks.split(',')
    tasks_df = pd.concat([ppt_df[ppt_df['task']==task] for task in tasks],axis=0)
else: 
    print('task type not accepted')
    quit()
print(tasks_df)
print(tasks_df.columns)

# fetch LLM
use_cuda = args.use_cuda
model_type, model_id = args.model_type, args.model_name_or_path
max_seq_len = args.max_seq_len

if use_cuda:
    device = 0
    dtype = torch.float16
else:
    device = torch.device('cpu')
    dtype = torch.float32

if model_type == 'huggingface':
    if re.search('t5-|alpaca|bart-', model_id):
        pipe_type = "text2text-generation"
    else:
        pipe_type = "text-generation"
    print(pipe_type)
    
    if 'llama' in model_id:
        from transformers import LlamaTokenizer, LlamaConfig
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        config = LlamaConfig.from_pretrained(model_id)
    else:
        from transformers import AutoTokenizer, AutoConfig
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
    hf_pipe = pipeline(pipe_type, model=model_id, tokenizer=tokenizer, device=device, torch_dtype=dtype)
    llm = hf_pipe#
elif model_type.startswith('openai'):
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY is None:
        API_KEY = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
        assert API_KEY.startswith("sk-"), "This doesn't look like a valid OpenAI API key"
        
        print("OpenAI API key configured")

    if model_type == 'openai':
        llm = OpenAI(model_name=model_id, temperature=0, openai_api_key=API_KEY)
        
    if model_type == 'openai_chat':
        llm = ChatOpenAI(model_name=model_id, temperature=0, openai_api_key=API_KEY)
else:
    print("Unsupported Model: {}".format(model_type))

data_name_or_path, data_split = args.data_name_or_path, args.data_split

# set result directory
res_path = os.path.join(args.result_path, '%s_res.tsv'%args.model_name_or_path.replace('/','-'))

# create empty dataframes to save results
res_df = pd.read_csv(res_path,sep='\t') if os.path.exists(res_path) else pd.DataFrame()
prev_tasks = set(res_df['task']) if len(res_df) else set()
tasks_df = tasks_df[~tasks_df['task'].isin(prev_tasks)]
print('%d tasks remaining'%len(tasks_df))

# for each task
for i,task_info in tqdm(tasks_df.iterrows()):
    task_info = dict(task_info)
    task, task_type = task_info['task'],task_info['type']
    print(task_info)
    
    # load dataset for task
    dataset = load_dataset(args.data_name_or_path, task, data_split)[data_split]
    
    if task_type == 'PAIR' or task_type == 'CLS':
        ppt_template = "%s\nOptions:\n%s\nPlease only answer with the options. "%(task_info['question'], '\n'.join(eval(task_info['options'])))
    else:
        ppt_template = "%s\n"%(task_info['question'])
    
    # specify instructions for alpaca or llama-2 models
    if re.search('alpaca|llama', model_id):
        ppt_template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
        \n\n### Instruction:\nYou will be presented with a question, please answer that correctly.\
        \n\n### Input:\n%s Provide the answer without explaining your reasoning. \n\n### Response:"%ppt_template
    else:
        ppt_template = "Question: %sAnswer:"%ppt_template
    print(ppt_template)
    
    ln_template = len(tokenizer.tokenize(ppt_template))
    valid_ln = max_seq_len - ln_template - args.max_new_tokens    
    
    prompts = []
    if task_type == 'PAIR':
        pairs = [it.split('[SEP]') for it in dataset['text']]
        for text_a, text_b in pairs:
            en_text_a, en_text_b = tokenizer.encode(text_a), tokenizer.encode(text_b)
            ln_a, ln_b = len(en_text_a), len(en_text_b)
            if (ln_a+ln_b)> valid_ln:
                if ln_a>ln_b:
                    en_text_a = en_text_a[:ln_a+ln_b-valid_ln]
                else:
                    en_text_b = en_text_b[:ln_a+ln_b-valid_ln]
            text_a, text_b = tokenizer.decode(en_text_a), tokenizer.decode(en_text_b)
            prompts.append(ppt_template.replace("{text_a}", text_a).replace("{text_b}", text_b))            
        
        # dataset = dataset.add_column('prompt', [ppt_template.replace("{text_a}", it[0]).replace("{text_b}", it[1]) for it in pairs])
    else:
        for text in dataset['text']:
            prompts.append(ppt_template.replace("{text}", truncate(text, tokenizer, valid_ln)))
        # dataset = dataset.add_column('prompt', [ppt_template.replace("{text}", truncate(it, args.max_length)) for it in dataset['text']])
    dataset = dataset.add_column('prompt', prompts)
        
    print(dataset['prompt'][:1])
    
    if task_type == 'PAIR' or task_type == 'CLS':
        d_labels = [it.replace('-', ' ').lower() for it in dataset.info.features['label'].names]
        labels = eval(task_info['options'])
        label2id = {l:i for i,l in enumerate(labels)}
        print(d_labels, labels, label2id)

    # optional: use only up to first 1000 samples (SOCKETTE) for quicker evaluation
    if args.unit_test:
        dataset = dataset[:args.batch_size]
    elif args.use_sockette:
        dataset = dataset[:1000]
    
    # iterate through batches to get prediction results    
    batch_size=int(args.batch_size)
    data_iter = data_iterator(dataset['prompt'], batch_size)
    outputs = []
    for batch in tqdm(data_iter, total=int(len(dataset['prompt'])/batch_size)):
        if pipe_type=='text-generation':
            output = llm(batch, max_new_tokens = args.max_new_tokens, return_full_text=False, clean_up_tokenization_spaces=True)
        elif pipe_type=='text2text-generation':
            output = llm(batch, max_new_tokens = args.max_new_tokens)
        elif pipe_type=='gpt':
            sys.exit(0) # later
        
        outputs.extend(output)
    
    # process prediction results
    dataset = pd.DataFrame(dataset)
    dataset['task'] = task
    outs = []
    for it in outputs:
        if pipe_type=='text-generation':
            if 'llama' in model_id:
                answer = ' '.join(it[0]['generated_text'].split()).strip()
            else:
                answer = it[0]['generated_text'].strip().split('\n')[0].strip()
        elif pipe_type=='text2text-generation':
            answer = ' '.join(it['generated_text'].split()).strip()
        outs.append(answer)
    
    dataset['generated_text'] = outs
    res_df = pd.concat([res_df, dataset])
    
    # save updated predictions
    res_df.to_csv(res_path,index=False,sep='\t')
    
