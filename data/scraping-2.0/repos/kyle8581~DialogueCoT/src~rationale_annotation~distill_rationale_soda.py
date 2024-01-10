

import numpy as np
from tqdm import tqdm
from langchain.llms import OpenAIChat, OpenAI
from langchain.chat_models import ChatOpenAI, openai
import json
import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio

import os
import evaluate
from datasets import load_dataset
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from pathlib import Path
import warnings
import logging
import random

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.WARNING)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)

warnings.filterwarnings("ignore")
openai.logger.propagate = False
import sys
logging.disable(sys.maxsize)
parser = argparse.ArgumentParser()

parser.add_argument("--split", type=str, choices=["train", "test","validation"])
parser.add_argument("--n_frag", type=int, default=100)
parser.add_argument("--frag_index", type=int)
parser.add_argument("--n_shot", type=int, default=5)
parser.add_argument("--n_sample", type=int, default=5)

args = parser.parse_args()

dataset = load_dataset("allenai/soda")
train_data = dataset[args.split]
random.seed(0)
random.shuffle(train_data)
save_dir = f"data/distilled/soda/frag_{args.n_frag}/{args.frag_index}"
if args.n_frag:
    frag_len = int(len(train_data)/args.n_frag)
    train_data = train_data[frag_len*args.frag_index:frag_len*(args.frag_index+1)]



with open(f"prompts/rg_focused_prompt/{args.n_shot}_shot.txt", "r") as f:
    prompt = f.read()
    

all_model_inputs = []
all_contexts = []
all_targets  = []

def add_name_tag_to_dialog(dialog):
    for i in range(len(dialog)):
        name_tag = "A: " if i%2==0 else "B: "
        dialog[i] = name_tag + dialog[i]
    return dialog

for dialogue in train_data['dialogue']:
    dialogue = add_name_tag_to_dialog(dialogue)
    if len(dialogue)<4: continue
    for cur_turn_idx in range(3,len(dialogue)):
        context = dialogue[:cur_turn_idx]
        target = dialogue[cur_turn_idx]
        if len(context) <=4: continue
        string_context = "\n".join(context)
        cur_example = f"\n{string_context}\nTarget:\n{target}\nRationale:\n"

        model_input = prompt +"\n"+cur_example
        all_model_inputs.append(model_input)
        all_contexts.append(context)
        all_targets.append(target)

print(f"Total instances: {len(all_model_inputs)}")


collected_predictions = []

total_cost = 0
lock = asyncio.Lock()
async def async_generate(llm, model_input, i):
    global total_cost
    global collected_predictions
    system_message = SystemMessage(content="Follow the instruction to generate rationales.")
    while True:
        try:
            response = await llm.agenerate([[system_message, HumanMessage(content=model_input)]] )
            token_used  = response.llm_output['token_usage']['total_tokens']
            total_cost += token_used / 1000 * 0.002
            print(total_cost)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None
    async with lock:
        collected_predictions.append({"context": all_contexts[i], "target": all_targets[i], "prediction": response.generations[0][0].text}) 
        try:
            if len(collected_predictions)%30 == 0:
                with open(save_dir+"_merged.json","w") as f:
                    json.dump(collected_predictions, f, indent=4)
        except:
            pass

async def generate_concurrently(all_model_input):
    llm = ChatOpenAI(temperature=0.5, max_tokens=300)
    tasks = [async_generate(llm, model_input, i) for i, model_input in enumerate(all_model_input) ]
    await tqdm_asyncio.gather(*tasks)
async def main():
    await generate_concurrently(all_model_inputs)
if __name__=="__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with DisableLogger():
        asyncio.run(main())
    with open(save_dir+"_merged.json","w") as f:
        json.dump(collected_predictions, f, indent=4)
