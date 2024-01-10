import numpy as np
from tqdm import tqdm
from langchain.llms import OpenAIChat, OpenAI
from langchain.chat_models import ChatOpenAI
import json
import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio
import time
import os
import random
# from nlgeval.pycocoevalcap.bleu.bleu import Bleu
# from nlgeval.pycocoevalcap.rouge.rouge import Rouge
# from nlgeval.pycocoevalcap.cider.cider import Cider
# from nlgeval.pycocoevalcap.meteor.meteor import Meteor
# from sentence_transformers import SentenceTransformer, util
import evaluate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--cs", choices=['dialect','ours', 'gpt_label', "None", "ours_last_inference", "gpt_last_inference","ours_only_answer", "reflect", "cot"], default=None)
parser.add_argument("--num_sample", type=int)
parser.add_argument("--min_context_len", type=int, default=0)
parser.add_argument("--min_target_len", type=int, default=0)
parser.add_argument("--n_shot", type=int, default=0)
parser.add_argument("--replace", type=str, default=None)
parser.add_argument("--prompt", type=str, default=None)

args = parser.parse_args()
# with open(f"prompts/ours/{relation_type}/{args.n_shot}_shot.txt", "r") as f:
if args.prompt:
    prompt_path = args.prompt
else:
    prompt_path = f"prompts/response_generation/{args.n_shot}_shot/{args.cs}.txt"
with open(prompt_path, "r") as f:
    prompt = f.read()
    


with open(args.input_file, "r") as f:
    train_data = json.load(f) 



def parse_data(dialogue, clip=True):
    target = dialogue['context'][dialogue['index']+1]
    if clip:
        context = dialogue['context'][:dialogue['index']+1]
    else:
        context = dialogue['context']
    id = dialogue['id']
    label_inference = dialogue['output']
    question = dialogue['input'].split("\n")[0].strip()
    
    return context, target, label_inference, question, id
total_cost = 0
collection_commonsense = []



    

# if split == "test": train_data = train_data[:int(len(train_data)*0.1)]

all_model_inputs = []
all_contexts = []
all_targets  = []
all_cs = []
total_cost = 0
async def async_generate(llm, model_input, i):
    global total_cost, all_cs
    system_message = SystemMessage(content="Follow the instruction to generate target response.")
    while True:
        try:
            response = await llm.agenerate([[system_message, HumanMessage(content=model_input)]] )
            token_used  = response.llm_output['token_usage']['total_tokens']
            total_cost += token_used / 1000 * 0.002
            if args.cs == "cot":
                if "Next Response:\n" in response.generations[0][0].text:
                    commonsense_knowledge, next_response = response.generations[0][0].text.split("Next Response:\n")
                    next_response = next_response.split(":")[-1].strip()
                else:
                    commonsense_knowledge = "\n".join(response.generations[0][0].text.split("\n")[:-1])
                    next_response = response.generations[0][0].text.split("\n")[-1].split(":")[-1].strip()
                all_cs[i] = commonsense_knowledge # save the rationale generated while cot 
            else:
                next_response = response.generations[0][0].text
            print(all_model_inputs[i])
            print(i, total_cost)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None
    with open(os.path.join(save_dir, f"{i}.json"),"w") as f:
        json.dump({"context": all_contexts[i], "target": all_targets[i], "cs": all_cs[i], "prediction": next_response}, f, indent=4)

async def generate_concurrently(all_model_input, start_idx):
    if args.cs == "cot":
        llm = ChatOpenAI(temperature=0.0, max_tokens=400, max_retries=100)
    else:
        llm = ChatOpenAI(temperature=0.0, max_tokens=100, max_retries=100)

    tasks = [async_generate(llm, model_input, i+start_idx) for i, model_input in enumerate(all_model_input) ]
    await tqdm_asyncio.gather(*tasks)
    

for d in train_data:
    context = d['context']
    if len(context)< args.min_context_len:
        continue
    if len(d['target'])< args.min_target_len:
        continue
    target = d['target']
    string_context = "\n".join(context)
    if args.cs == "None":
        commonsense_knowledge = ""
    elif args.cs =="reflect":
        commonsense_knowledge = d['prediciton']
    elif args.cs == "gpt_label":
        commonsense_knowledge = d['gpt_label']
    elif args.cs == "gpt_last_inference":
        commonsense_knowledge  = d['gpt_label'].split("\n")[-1]
    elif args.cs == "ours_last_inference":
        commonsense_knowledge = d['prediction'].split("\n")[-1]
        if "Subanswer" in commonsense_knowledge:
            commonsense_knowledge = commonsense_knowledge.split(":")[-1].strip(" ")
    elif args.cs == "ours_only_answer":
        commonsense_knowledge = d['prediction']
        if "Subquestion" not in d['prediction']:
            commonsense_knowledge = "None"
        else:
            commonsense_knowledge = "\n".join([a for a in d["prediction"].split("\n") if "Subanswer" in a])
    elif args.cs == "cot":
        commonsense_knowledge = ""
    else:
        commonsense_knowledge = d['prediction']
        if "Subquestion" not in d['prediction']:
            commonsense_knowledge = "None"
    
    name_tag = target[:target.index(":")+1]
    # cur_example = f"\nRationale:\n{commonsense_knowledge}\n{string_context}\nTarget:\n"

    model_input = prompt.format(**locals())
    all_model_inputs.append(model_input)
    all_contexts.append(context)
    all_targets.append(target[target.index(":")+1:])
    all_cs.append(commonsense_knowledge)

random.seed(0)
if args.num_sample:
    cand_indices = [i for i in range(len(all_model_inputs))]
    sampled_indices = random.sample(cand_indices, args.num_sample)

    all_model_inputs = [all_model_inputs[i] for i in sampled_indices]
    all_contexts = [all_contexts[i] for i in sampled_indices]
    all_targets = [all_targets[i] for i in sampled_indices]
    all_cs = [all_cs[i] for i in sampled_indices]

async def main(model_inputs, start_idx):
    await generate_concurrently(model_inputs, start_idx)

if __name__=="__main__":
    print(all_model_inputs[0])
    save_dir = args.save_dir
    save_dir +=  f"_num_sample_{args.num_sample}_min_turn_{args.min_context_len}_min_target__{args.min_target_len}"
    def save_and_calc_metric():
        separte_files = os.listdir(save_dir)
        annotated_data = [json.load(open(os.path.join(save_dir, fn),"r")) for fn in separte_files ]
            
        all_predictions = [d['prediction'].lower() for d in annotated_data]
        references = [d['target'].lower() for d in annotated_data]
        all_metric = {}
        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(predictions=all_predictions, references=references, use_aggregator=False)
        for k,v in rouge_score.items():
            all_metric[k] = str(round(np.mean(v),4))
            for i in range(len(annotated_data)):
                annotated_data[i][k] = v[i]
        # print(rouge_score)
        bleu = evaluate.load("bleu")
        for i in range(1,5):
            bleu_score = bleu.compute(predictions=all_predictions, references=references, max_order=i)['bleu']
            all_metric[f"bleu{i}"] = str(round(bleu_score,4))
        ##########################
        # SAVE PREDICTION
        #########################
        with open(save_dir+"merged.json","w") as f:
            json.dump(annotated_data, f, indent=4)
        bertscore = evaluate.load("bertscore")
        bertscore_score = round(np.mean(bertscore.compute(predictions=all_predictions, references=references, lang="en")['f1']),4)
        all_metric["bertscore"] = str(bertscore_score)
        # print(bertscore_score)
        with open(save_dir+"merged.csv","w") as f:
            f.write(",".join(list(all_metric.keys()))+"\n")
            f.write(",".join(list(all_metric.values())))
    if args.replace and os.path.exists(save_dir):
        answer = input(f"Are you sure you want to remove directory {save_dir}?(y/n)")
        if answer == "y":
            os.removedirs(save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if len(all_model_inputs) > 300:
            for start_idx in tqdm(range(0,len(all_model_inputs), 300)):
                cur_model_inputs = all_model_inputs[start_idx:start_idx+300]

                asyncio.run(main(cur_model_inputs, start_idx))
                save_and_calc_metric()
        else:
            asyncio.run(main(all_model_inputs,0))
            
    save_and_calc_metric()
    print(all_model_inputs[0])
    
    