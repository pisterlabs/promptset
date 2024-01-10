import json
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
sys.path.insert(1, '../utils')
from text_utils import TextEncoder, fix_malformed
import pickle
import numpy as np
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from modeling_openai import OpenAIGPTMemModel
from decoding import beam_search 
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_multigpu',type=bool,default=True)
    parser.add_argument('--n_gpu',type=int,default=1)
    parser.add_argument('--load_epoch',type=str,default='1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kg_type',type=str,default='atomic')
    parser.add_argument('--use_mem',type=bool,default=False)
    parser.add_argument('--comet',type=bool, default=False)
    parser.add_argument('--gen_len',type=int, default=50)
    parser.add_argument('--model_type',type=str,default='./models/model/') #specify model path
    parser.add_argument('--save_filename',type=str,default='outputs.jsonl')
    parser.add_argument('--save_dir',type=str,default='../../data/gen_data/gpt')
    parser.add_argument('--original_file',type=str, default='examples.jsonl')
    parser.add_argument('--data_dir',type=str,default='../../data')
    parser.add_argument('--n_batch',type=int,default=1)
    parser.add_argument('--beam',type=int,default=10)
    parser.add_argument('--filter_decode',type=bool,default=True)
    parser.add_argument('--mem_k',type=int,default=1)
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_mem = args.use_mem
device = torch.device(device)
text_encoder = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
encoder = text_encoder.encoder
decoder = text_encoder.decoder

#sentence-level special tokens
encoder['<|sent0|>'] = len(encoder)
decoder[len(decoder)] = '<|sent0|>'

encoder['<|sent1|>'] = len(encoder)
decoder[len(decoder)] = '<|sent1|>'

encoder['<|sent2|>'] = len(encoder)
decoder[len(decoder)] = '<|sent2|>'

encoder['<|sent3|>'] = len(encoder)
decoder[len(decoder)] = '<|sent3|>'

encoder['<|sent4|>'] = len(encoder)
decoder[len(decoder)] = '<|sent4|>'

sent_ids = [encoder['<|sent0|>'],encoder['<|sent1|>'],encoder['<|sent2|>'],encoder['<|sent3|>'],encoder['<|sent4|>']]

#ATOMIC special tokens
encoder['<|xNeed|>'] = len(encoder)
decoder[len(decoder)] = '<|xNeed|>'

encoder['<|xIntent|>'] = len(encoder)
decoder[len(decoder)] = '<|xIntent|>'

encoder['<|xWant|>'] = len(encoder)
decoder[len(decoder)] = '<|xWant|>'

encoder['<|oEffect|>'] = len(encoder)
decoder[len(decoder)] = '<|oEffect|>'

encoder['<|xReact|>'] = len(encoder)
decoder[len(decoder)] = '<|xReact|>'

encoder['<|oWant|>'] = len(encoder)
decoder[len(decoder)] = '<|oWant|>'

encoder['<|oReact|>'] = len(encoder)
decoder[len(decoder)] = '<|oReact|>'

encoder['<|xEffect|>'] = len(encoder)
decoder[len(decoder)] = '<|xEffect|>'

encoder['<|xAttr|>'] = len(encoder)
decoder[len(decoder)] = '<|xAttr|>'

encoder['<|PAD|>'] = len(encoder)
decoder[len(decoder)] = '<|PAD|>'

encoder['<|endoftext|>'] = len(encoder) 
decoder[len(decoder)] = '<|endoftext|>'

text_encoder.encoder = encoder
text_encoder.decoder = decoder
n_vocab = len(text_encoder.encoder)

best_model = 'best_params_' + args.load_epoch
model_path = os.path.join(args.model_type,best_model)

if use_mem or 'mem' in args.model_type:
   model = OpenAIGPTMemModel.from_pretrained('openai-gpt')
else:
   model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.resize_token_embeddings(n_vocab)
if args.use_multigpu:
   model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
model.eval()

print("device", device, "n_gpu", args.n_gpu)
gen_len = args.gen_len

def get_token(next_idx, tokenizer=text_encoder):
    try:
       return tokenizer.decoder[next_idx]
    except:
       return next_idx

if not os.path.exists(args.save_dir):
   os.makedirs(args.save_dir)
gen_file = open(os.path.join(args.save_dir, 'beam_' + args.save_filename),'w')

dims_ = {'atomic':["<|xNeed|>","<|xIntent|>","<|xWant|>","<|oEffect|>","<|xReact|>","<|oWant|>","<|oReact|>","<|xEffect|>","<|xAttr|>"]}[args.kg_type]
dims = [encoder[d] for d in dims_]

def clean_gen(gen):
    gen = [w for w in gen.tolist() if w != encoder['<|PAD|>']]
    gen = [get_token(idx) for idx in gen]
    if '<unk>' in gen:
       gen = [t for t in gen if t != '<unk>']
    gen = "".join([word.replace("</w>", " ") for word in gen])
    gen = gen.replace("<|endoftext|>","")
    if len(gen) > 0 and gen[-1] == ' ':
       gen = gen[:-1]
    return fix_malformed(gen)

def pad_rels(relation, pad_len=100):
    return relation[:100] + [encoder['<|PAD|>']] * (100-len(relation[:100]))

if use_mem:
   external_mem = {}

n_updates = 0 
for line_ in [json.loads(l) for l in open(args.original_file).readlines()]:
    id = line_["storyid"]
    if use_mem:
       if id not in external_mem.keys():
          external_mem[id] = []
          size_mem = 0
       else:
          continue 
    original = [line_['sentence1'],line_['sentence2'],line_['sentence3'],line_['sentence4'],line_['sentence5']]
    save_output = {}
    save_output["storyid"] = id 
    save_output["story"] = " ".join(original)
    save_output["gold_relations"] = line_["distance_supervision_relations"]
    for sent_id in sent_ids:
        with torch.no_grad():
             for d in range(len(dims)):
                 XMB = [encoder['<|PAD|>']] * 500
                 if args.filter_decode and ('xIntent' in dims_[d] or 'xNeed' in dims_[d] or 'xAttr' in dims_[d]):
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original[:int(decoder[sent_id].split('<|sent')[1].replace('|>',''))+1])))
                 else:
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original)))
                 context += [sent_id, dims[d]]
                 i_1 = len(context)-1
                 XMB[:len(context)] = context 
                 XMB = torch.tensor(XMB,dtype=torch.long).to(device)
                 XMB = XMB.unsqueeze(0)
                 if use_mem and size_mem != 0:
                    mem = external_mem[id]
                    mem = torch.LongTensor([pad_rels(r) for r in mem]).unsqueeze(0)
                    gen = beam_search(model, encoder, XMB,i_1,mem=mem,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)
                 else:
                    if use_mem:
                       gen = beam_search(model, encoder, XMB,i_1,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)                          
                    else:
                       gen = beam_search(model, encoder, XMB, i_1,num_beams=args.beam)
                 gen = [clean_gen(g) for g in gen]
                 if use_mem:
                    mem_gen = gen[0]
                    size_mem += 1
                    external_mem[id].append(text_encoder.convert_tokens_to_ids(text_encoder.tokenize(mem_gen)))
                 if decoder[sent_id] + '_' + "generated_relations" in save_output.keys(): 
                    save_output[decoder[sent_id] + '_' + "generated_relations"].append(gen)
                    save_output[decoder[sent_id] + '_' + "generated_dims"].append([decoder[dims[d]]] * len(gen))
                 else:
                    save_output[decoder[sent_id] + '_' + "generated_relations"] = [gen]
                    save_output[decoder[sent_id] + '_' + "generated_dims"] = [[decoder[dims[d]]] * len(gen)]
    gen_file.write(json.dumps(save_output) + '\n')
    n_updates += 1
