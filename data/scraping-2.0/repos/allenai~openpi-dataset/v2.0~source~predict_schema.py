import argparse
import openai
import json
import random
random.seed(299)
from sklearn.metrics import accuracy_score
import backoff

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='davinci', type=str, help='Either davinci or chatgpt.')
parser.add_argument('--key', default='harry_ccbft', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--seed', default='', type=str, help='Random seed.')
parser.add_argument('--split', default='dev', type=str, help='The split to evaluate on.')
parser.add_argument('--prompt', default='1', type=str, help='Type of prompt.')
parser.add_argument('--parse', action='store_true', help='Whether to parse the type 2 output.')

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
            
def apply_fewshot_template(examples):
    template = ""
    for example in examples:
        template += f"""A person's goal is to {example["goal"].lower()}.
For each of the steps, list the involved entities and attributes THAT UNDERGO ANY CHANGE. For example, for the step 'heat the oven', rack (temperature) is correct, while oven (color) is wrong. 
"""
        for i, (step, e_a) in enumerate(example["gold_step_entities_attributes"].items()):
            template += f"""Step: {example["steps"][i]}
Entities and attributes: """
            for entity, attributes in e_a.items():
                entity = entity.split(' | ')[0]
                attributes = [a[0].split(' | ')[0] for a in attributes]
                template += entity + " (" + ','.join(attributes) + '), '
            template += "\n"
        template += "\n"
    #print(template)
    #raise SystemExit
    return template

def apply_inference_template(example, previous_outputs=[]):
    template = f"""A person's goal is to {example["goal"].lower()}.
For each of the steps, list the involved entities and attributes THAT UNDER GO ANY CHANGE. For example, for the step 'heat the oven', rack (temperature) is correct, while oven (color) is wrong.
"""
    template += f"""Step: {example["steps"][0]}
Entities and attributes:"""
    for i,previous_output in enumerate(previous_outputs):
        template += ' ' + previous_output + '\n'
        template += f"""Step: {example["steps"][i+1]}
Entities and attributes:"""
    return template

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

def apply_inference_template_2(example, previous_outputs=[]):
    template = f"""A person's goal is to {example["goal"].lower()}.
For each of the steps, list all the state changes of involved entities and attributes.
"""
    template += f"Step: {example['steps'][0]}"
    for i,previous_output in enumerate(previous_outputs):
        template += ' ' + previous_output + '\n'
        template += f"Step: {example['steps'][i+1]}"
    #print(template)
    #raise SystemExit
    return template

def apply_fewshot_template_chatgpt(examples):
    template = []
    template.append({"role": "system", "content": "You are a helpful assistant that figures out involved entities and attributes in procedures."})
    for example in examples:
        template.append({"role": "user", "content": f"A person's goal is to " + example["goal"].lower() + ". For each of the steps, you will list entities and attributes THAT UNDER GO ANY CHANGE. For example, for the step 'heat the oven', rack (temperature) is a good answer, while oven (color) is a bad answer. Are you ready?"})
        template.append({"role": "assistant", "content": "Yes, I'm ready."})
        for i, (step, e_a) in enumerate(example["gold_step_entities_attributes"].items()):
            template.append({"role": "user", "content": f"Step: " + example["steps"][i]})
            response_str = ""
            for entity, attributes in e_a.items():
                entity = entity.split(' | ')[0]
                attributes = [a[0].split(' | ')[0] for a in attributes]
                response_str += entity + " (" + ','.join(attributes) + '), '
            template.append({"role": "assistant", "content": response_str})
    template += [{"role": "user", "content": "Next, I will provide you with another procedure. Please answer in the exact same format as before. OK?"}, {"role": "assistant", "content": "Yes, please go ahead."}]      
    #print(template)
    #raise SystemExit
    return template

def apply_inference_template_chatgpt(example, previous_outputs=[]):
    template= [{"role": "user", "content": "A person's goal is to " + example["goal"].lower() + ". For each of the steps, you will list the involved entities and attributes. Answer in the format of 'entity1 (attribute1.1,attribute1.2), entity2 (attribute2)' and so on. Are you ready?"}]
    template.append({"role": "assistant", "content": "Yes, I'm ready."})
    template.append({"role": "user", "content": f"Step: " + example["steps"][0]})
    for i,previous_output in enumerate(previous_outputs):
        template.append(previous_output)
        template.append({"role": "user", "content": f"Step: " + example["steps"][i+1]})
    return template

def apply_fewshot_template_chatgpt_2(examples):
    template = []
    template.append({"role": "system", "content": "You are a helpful assistant that figures out state changes of involved entities and attributes in procedures."})
    for example in examples:
        template.append({"role": "user", "content": f"A person's goal is to " + example["goal"].lower() + ". For each of the steps, you will list all state changes of entities and attributes. You will answer in this format:\n  - attribute_name of entity_name was before_state before and after_state after\n For example:\n  - temperature of oven was cool before and hot afterwards.\nAre you ready?"})
        template.append({"role": "assistant", "content": "Yes, I'm ready."})
        for i, (step, e_a) in enumerate(example["gold_step_entities_attributes"].items()):
            template.append({"role": "user", "content": f"Step: " + example["steps"][i]})
            response_str = ""
            for entity, attributes in e_a.items():
                    for attribute, pre, post in attributes:
                        response_str += f"  - {attribute.split(' | ')[0]} of {entity.split(' | ')[0]} was {pre.split(' | ')[0]} before and {post.split(' | ')[0]} after\n"
            template.append({"role": "assistant", "content": response_str})
    template += [{"role": "user", "content": "Next, I will provide you with another procedure. Please answer in the exact same format as before. OK?"}, {"role": "assistant", "content": "Yes, please go ahead."}]      
    #print(template)
    #raise SystemExit
    return template

def apply_inference_template_chatgpt_2(example, previous_outputs=[]):
    template= [{"role": "user", "content": f"A person's goal is to " + example["goal"].lower() + ". For each of the steps, you will list all state changes of entities and attributes. You will answer in this format:\n  - attribute_name of entity_name was before_state before and after_state after\n For example:\n  - temperature of oven was cool before and hot afterwards.\nAre you ready?"}]
    template.append({"role": "assistant", "content": "Yes, I'm ready."})
    template.append({"role": "user", "content": f"Step: " + example["steps"][0]})
    for i,previous_output in enumerate(previous_outputs):
        print(i)
        template.append(previous_output)
        template.append({"role": "user", "content": f"Step: " + example["steps"][i+1]})
    return template

def build_fewshot(model):
    # Randomly choose 5 procs from train
    train_examples = parse_data("train")

    if model == "davinci":
        NUM_SHOTS = 1
        #selected_examples = random.sample(train_examples, NUM_SHOTS)
        selected_examples = [train_examples[192]]
        if args.prompt == "1":
            fewshot = apply_fewshot_template(selected_examples)
        elif args.prompt == "2":
            fewshot = apply_fewshot_template_2(selected_examples)
    elif model == "chatgpt":
        NUM_SHOTS = 1
        #selected_examples = random.sample(train_examples, NUM_SHOTS)
        selected_examples = [train_examples[192]]
        if args.prompt == "1":
            fewshot = apply_fewshot_template_chatgpt(selected_examples)
        elif args.prompt == "2":
            fewshot = apply_fewshot_template_chatgpt_2(selected_examples)
    #print(fewshot)
    return fewshot

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
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

def predict():
    examples = parse_data(args.split)
    prompt_fewshot = build_fewshot(args.model)
    if args.model == "davinci":
        if args.prompt == "1":
            apply_template = apply_inference_template
            run = run_gpt
            stop = ['\n']
        elif args.prompt == "2":
            apply_template = apply_inference_template_2
            run = run_gpt
            stop = ['Step:']
    elif args.model == "chatgpt":
        if args.prompt == "1":
            apply_template = apply_inference_template_chatgpt
            run = run_chatgpt
            stop = []
        elif args.prompt == "2":
            apply_template = apply_inference_template_chatgpt_2
            run = run_chatgpt
            stop = []
    out_dict = {}

    for example in examples:
        out_dict[example["id"]] = []
        previous_outputs = []
        #out_dict[example[]]
        for _ in example["steps"]:
            #print(example)
            prompt = prompt_fewshot + apply_template(example, previous_outputs)
            print(prompt)
            #raise SystemExit
            if args.model == "davinci":
                output = run(prompt, stop=stop)
            elif args.model == "chatgpt":
                output = run(prompt)
            previous_outputs.append(output)
            #print(previous_outputs)
            #raise SystemExit

            # parse output
            output_str = output if args.model == "davinci" else output['content']
            pred_entities = []
            pred_entities_attributes = []
            #print(output_str)

            if args.prompt == "1":
                for e_a in output_str.split('), '):
                    try:
                        entity = e_a.split(" (")[0]
                        pred_entities.append(entity)
                        for attribute in e_a.split(" (")[1].split(','):
                            processed_attribute = attribute.strip().strip('.').strip(')')
                            if processed_attribute:
                                pred_entities_attributes.append((entity, processed_attribute))
                    except IndexError:
                        continue
            elif args.prompt == "2":
                for line in output_str.strip().split('\n'):
                    line = line.strip('  - ')
                    if args.parse:
                        be_word = " were " if " were " in line else " was "
                        try:
                            attribute = line.split(' of ')[0]
                        except:
                            attribute = "ERROR"
                        try:
                            entity = line.split(' of ')[1].split(be_word)[0]
                        except:
                            entity = "ERROR"
                        try:
                            pre = line.split(' of ')[1].split(be_word)[1].split(' before and ')[0]
                        except:
                            pre = "ERROR"
                        try:
                            post = line.split(' of ')[1].split(be_word)[1].split(' before and ')[1].split(' after')[0]
                        except:
                            post = "ERROR"
                        pred_entities_attributes.append((entity, attribute, pre, post))
                        print(entity, attribute, pre, post)
                    else:
                        pred_entities_attributes.append(line)
                        print(line)
            
            out_dict[example["id"]].append(pred_entities_attributes)

    return out_dict
        

        

def evaluate():
    with open(f'pred_{args.model}_{args.prompt}_{args.max_prompt}{args.seed}.txt', 'r') as f:
        preds = [x.strip() for x in f.readlines()]
    with open('gold.txt', 'r') as f:
        golds = [x.strip() for x in f.readlines()]
    print("Accuracy", accuracy_score(golds, preds))
    return "Accuracy", accuracy_score(golds, preds)

if __name__ == "__main__":
    out_dict = predict()
    pred_data = {}
    if args.prompt == "1":
        for id, proc in out_dict.items():
            pred_data[id] = []
            d = {}
            for step in proc:
                for e, a in step:
                    if e not in d:
                        d[e] = [a]
                    else:
                        d[e].append(a)
                pred_data[id].append(d)
                d = {}
    else:
        pred_data = out_dict
    with open(f"../data/{args.split}_schema_{args.model}_{args.prompt}.json", "w") as f:
        json.dump(pred_data, f, indent=4)
