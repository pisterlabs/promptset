"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import openai
import random
from typing import List

openai.api_key = ''
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 50 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 2 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    leaderboard = {'Sampling': 0, 'A*': 0, 'MCTS': 0}
    with ctx:
        for k in range(num_samples):
            y_0 = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y_1 = model.astar_generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y_2 = model.mcts_generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            outputs = [decode(y_0[0].tolist()), decode(y_1[0].tolist()), decode(y_2[0].tolist())]
            print(outputs)
            oracle_input = "Rank the following model outputs based on how closely they resemble Shakespearean text."
            # Shuffle the order of the outputs and append them to the oracle input, labeling each one with a letter
            random_indexes = random.sample(range(3), 3)
            real_indexes = ['Sampling', 'A*', 'MCTS']
            alphabet = ['A', 'B', 'C']
            mapping = {}
            for i in range(len(random_indexes)):
                mapping[alphabet[i]] = real_indexes[random_indexes[i]]
            oracle_input += f"Completion A: {outputs[random_indexes[0]]}\n"
            oracle_input += f"Completion B: {outputs[random_indexes[1]]}\n"
            oracle_input += f"Completion C: {outputs[random_indexes[2]]}\n"
            oracle_input += "Based on the responses, strictly output a list that ranks responses from 'A', 'B', 'C' from best to worst based on what you think best answers the question(s). Here is an example ranking output: ['A', 'C', 'B']. You should only return a list in this format and NOTHING ELSE."
            # Query the oracle model
            oracle_msgs = [{"role": "user", "content": oracle_input}]
            oracle_response = openai.ChatCompletion.create(model='gpt-4', messages=oracle_msgs)
            best_response_evaluation = oracle_response["choices"][0]["message"]["content"].strip()
            print(best_response_evaluation)
            raw_ranks = str(oracle_response["choices"][0]["message"]["content"].strip())
            print(raw_ranks)
            res = raw_ranks.strip('][').split(', ')
            for k in range(len(res)):
                res[k] = res[k].replace('\'', '')
            awards = {0: 3, 1: 2, 2: 1}
            for j in range(len(res)):
                leaderboard[mapping[res[j]]] += awards[j]
            print("Current Leaderboard", leaderboard)


