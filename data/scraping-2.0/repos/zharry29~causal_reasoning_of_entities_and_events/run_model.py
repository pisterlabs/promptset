import argparse
import openai
import csv
import random
random.seed(29)
import backoff
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='code-davinci-002', type=str, help='Model ID from OpenAI.')
parser.add_argument('--path', required=True, type=str, help='Path to the folder containing the setting.')
parser.add_argument('--hop', default='1', type=str, help='Whether the prompt is 0-hop+1-hop or 1-hop only.')
parser.add_argument('--key', default='harry', type=str, help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../_private/{args.key}.key').read()

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError))
def run_llm(prompt, temperature=0, stop=['\n']):
    ret = openai.Completion.create(
        engine=args.model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop
    )

    gen_text = ret["choices"][0]["text"].strip()
    return gen_text

# from gpt3.niket1 import build_prompt
build_prompt = __import__(args.path.replace('/','.'))

if args.hop == '1':
    prompt_file_name = args.path + "/prompts/1hop_only.txt"
elif args.hop == '0':
    prompt_file_name = args.path + "/prompts/1hop_0hop.txt"
with open(prompt_file_name) as f:
    written_prompt = f.read()

with open('data_dev_v2.json') as f:
    jobj = json.load(f)

all_gold_changes = []
all_pred_changes = []

for id,v in jobj.items():
    goal = v['goal']
    steps = []
    events = []
    entities = []
    gold_event_changes = []
    gold_entity_changes = []
    for w in v['steps'][1:]:
        current_event_changes = []
        current_entity_changes = []
        for u in w:
            if u["type"] == "multihop":
                if u["event"] not in events:
                    events.append(u["event"])
                event_index = events.index(u["event"])
                current_event_changes.append((u["event"], u["change"].split()[0]))
            if u["type"] == "entity":
                if u["entity"] not in entities:
                    entities.append(u["entity"])  
                current_entity_changes.append(u)
            if u["type"] == "step":
                steps.append(u["step"])
        gold_event_changes.append(current_event_changes)
        gold_entity_changes.append(current_entity_changes)

    #print(gold_event_changes)
    pred_changes = eval(args.prompt)(goal, steps, entities, events, gold_entity_changes)
    #print(pred_changes)
    #print(len(gold_event_changes))
    #print(len(pred_changes))
    #assert(len(gold_event_changes) == len(pred_changes))
    if(len(gold_event_changes) < len(pred_changes)):
        print("Pred changes are more than gold, truncating.")
        pred_changes = pred_changes[:len(gold_event_changes)]
    all_gold_changes += gold_event_changes
    all_pred_changes += pred_changes
    #break

pred_all, pred_correct, gold_all, gold_correct = 0,0,0,0
assert(len(all_gold_changes) == len(all_pred_changes))
for gold_step_changes, pred_step_changes in zip(all_gold_changes, all_pred_changes):
    for gold_step_change in gold_step_changes:
        gold_all += 1
        if gold_step_change in pred_step_changes:
            gold_correct += 1
    for pred_step_change in pred_step_changes:
        pred_all += 1
        if pred_step_change in gold_step_changes:
            pred_correct += 1

precision = pred_correct/pred_all
recall = gold_correct/gold_all
f1 = 2 * precision * recall / (precision + recall)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
