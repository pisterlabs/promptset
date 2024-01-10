import argparse
import openai
import json
import random
random.seed(299)
from sklearn.metrics import accuracy_score
import backoff

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='Either davinci or chatgpt.')
parser.add_argument('--key', default='harry_ccbft', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--seed', default='', type=str, help='Random seed.')
parser.add_argument('--split', default='dev', type=str, help='The split to evaluate on.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()
if args.seed:
    random.seed(int(args.seed[1:]))

def parse_data(split):
    parsed_examples = []
    with open(f'../data/{split}-ranked.json') as f:
        for id, proc in json.load(f).items():
            goal = proc["goal"]
            steps = proc["steps"]
            states = proc["states"]
            gold_step_entities_attributes = {f"step{i}": {} for i in range(1,len(steps)+1)}
            for state in states:
                entity = state["entity"]
                for step, answer in state["answers"].items():
                    if answer:
                        gold_step_entities_attributes[step][entity] = []
                        for att in answer:
                            gold_step_entities_attributes[step][entity].append((att["attribute"], att["before"], att["after"]))
            parsed_examples.append({
                "id": id,
                "goal": goal,
                "steps": steps,
                "gold_step_entities_attributes": gold_step_entities_attributes,
            })
    #print(parsed_examples[0])
    return parsed_examples

def apply_fewshot_template_2(examples):
    template = ""
    for example in examples:
        template += f"""A person's goal is to {example["goal"].lower()}.
For each of the steps, list all the state changes of involved entities and attributes.
"""
        for i, (step, e_a) in enumerate(example["gold_step_entities_attributes"].items()):
            template += f"Step: {example['steps'][i]}"
            for entity, attributes in e_a.items():
                for attribute, pre, post in attributes:
                    template += f"\n  - {attribute.split(' | ')[0]} of {entity.split(' | ')[0]} was {pre.split(' | ')[0]} before and {post.split(' | ')[0]} after"
            template += "\n"
        template += "\n"
    #print(template)
    #raise SystemExit
    return template

def apply_fewshot_template_chatgpt_2(examples):
    template = []
    template.append({"role": "system", "content": "You are a helpful assistant that predictes state changes entities and attributes in procedures."})
    for example in examples:
        template.append({"role": "user", "content": f"A person's goal is to {example['goal'].lower()}. Next, I'll provide you with a step and an attribute of an entity. You will return the states before and after doing this step. Your format would be\nBefore: some state\nAfter: some state\nIs that clear?"})
        template.append({"role": "assistant", "content": "Yes, I understand. Please go ahead."})
        for i, (step, e_a) in enumerate(example["gold_step_entities_attributes"].items()):
            for entity, attributes in e_a.items():
                for attribute, pre, post in attributes:
                    template.append({"role": "user", "content": f"Step: {example['steps'][i]}\nHow does the {attribute.split(' | ')[0]} of {entity.split(' | ')[0]} change?"})
                    template.append({"role": "assistant", "content": f"Before: {pre.split(' | ')[0]}\nAfter: {post.split(' | ')[0]}"})
    #print(template)
    #raise SystemExit
    return template

def build_fewshot(model):
    # Randomly choose 1 proc from train
    train_examples = parse_data("train")

    if model == "davinci":
        NUM_SHOTS = 1
        #selected_examples = random.sample(train_examples, NUM_SHOTS)
        selected_examples = [train_examples[192]]
        fewshot = apply_fewshot_template_2(selected_examples)
    elif model == "chatgpt":
        NUM_SHOTS = 1
        #selected_examples = random.sample(train_examples, NUM_SHOTS)
        selected_examples = [train_examples[192]]
        fewshot = apply_fewshot_template_chatgpt_2(selected_examples)
    #print(fewshot)
    return fewshot

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def run_gpt(prompt, model="text-davinci-003", temperature=0.5, stop=['\n']):
    ret = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop
    )
    gen_text = ret["choices"][0]["text"].strip()#.split('\n')[0]
    return gen_text

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def run_chatgpt(prompt, model="gpt-3.5-turbo", temperature=0.7):
    ret = openai.ChatCompletion.create(
        model=model,
        messages=prompt
        )
    gen_text = dict(ret["choices"][0]["message"])
    return gen_text

def predict_davinci():
    examples = parse_data(args.split)
    prompt_fewshot = build_fewshot(args.model)

    pred_dict = {}
    gold_dict = {}

    for example in examples:
        pred_dict[example["id"]] = []
        gold_dict[example["id"]] = []
        prompt = prompt_fewshot + f"""A person's goal is to {example["goal"].lower()}.
For each of the steps, list all the state changes of involved entities and attributes."""
        for i, step_block in enumerate(example["gold_step_entities_attributes"].values()):
            prompt += f"\nStep: {example['steps'][i]}"
            step_gold = []
            step_pred = []
            for entity, attributes_blocks in step_block.items():
                for attribute, pre, post in attributes_blocks:
                    prompt += f"\n  - {attribute.split(' | ')[0]} of {entity.split(' | ')[0]} was"
                    step_gold.append((entity.split(' | ')[0], attribute.split(' | ')[0], pre.split(' | ')[0], post.split(' | ')[0]))

                    #print(prompt)
                    #raise SystemExit
                    output = run_gpt(prompt, stop=['\n'])
                    prompt += ' ' + output
                    #print(output)
                    #raise SystemExit

                    # parse output
                    output_str = output if args.model == "davinci" else output['content']
                    #print(output_str)

                    pred_pre = output_str.strip().split(' before and ')[0]
                    pred_post = output_str.strip().split(' before and ')[1].split(' after')[0]
                    step_pred.append((entity, attribute, pred_pre, pred_post))
            
            pred_dict[example["id"]].append(step_pred)
            gold_dict[example["id"]].append(step_gold)

    return pred_dict, gold_dict

def predict_chatgpt():
    examples = parse_data(args.split)
    prompt_fewshot = build_fewshot(args.model)

    pred_dict = {}
    gold_dict = {}

    for example in examples:
        pred_dict[example["id"]] = []
        gold_dict[example["id"]] = []
        prompt = prompt_fewshot.copy()
        prompt.append({"role": "user", "content": f"A person's goal is to {example['goal'].lower()}. Next, I'll provide you with a step and an attribute of an entity. You will return the states before and after doing this step. Your format would be\nBefore: some state\nAfter: some state\nIs that clear?"})
        prompt.append({"role": "assistant", "content": "Yes, I understand. Please go ahead."})
        for i, step_block in enumerate(example["gold_step_entities_attributes"].values()):
            print(i)
            step_gold = []
            step_pred = []
            for entity, attributes_blocks in step_block.items():
                for attribute, pre, post in attributes_blocks:
                    new_prompt = prompt.copy()
                    new_prompt.append({"role": "user", "content": f"Step: {example['steps'][i]}\nHow does the {attribute.split(' | ')[0]} of {entity.split(' | ')[0]} change?"})
                    step_gold.append((entity.split(' | ')[0], attribute.split(' | ')[0], pre.split(' | ')[0], post.split(' | ')[0]))

                    #print(new_prompt)
                    #raise SystemExit
                    output = run_chatgpt(new_prompt)
                    # parse output
                    output_str = output['content']
                    print(output_str)
                    #prompt.append({"role": "assistant", "content": output_str})
                    #print(output)
                    #raise SystemExit

                    try:
                        pred_pre = output_str.strip().split('Before: ')[1].split("\nAfter: ")[0]
                        pred_post = output_str.strip().split("\nAfter: ")[1]
                    except:
                        pred_pre = "Error"
                        pred_post = "Error"
                    step_pred.append((entity, attribute, pred_pre, pred_post))
                    #print(pred_pre, pred_post)
                    #raise SystemExit
            
            pred_dict[example["id"]].append(step_pred)
            gold_dict[example["id"]].append(step_gold)

    return pred_dict, gold_dict

if args.model == "davinci":
    pred_dict, gold_dict = predict_davinci()
elif args.model == "chatgpt":
    pred_dict, gold_dict = predict_chatgpt()
with open(f"../data/{args.split}_states_{args.model}.json", "w") as f:
    json.dump(pred_dict, f, indent=4)

with open(f"../data/{args.split}_states_gold.json", "w") as f:
    json.dump(gold_dict, f, indent=4)