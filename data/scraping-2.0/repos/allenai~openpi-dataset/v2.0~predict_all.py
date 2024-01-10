# An API that takes in a procedure and predicts
# - the schema (entities and attributes that undergo changes)
# - the values of these changes
# - the salience of these entities

import json
import random
import openai
import backoff
import argparse
import pickle
import re

random.seed(299)

parser = argparse.ArgumentParser()
parser.add_argument('--key', default='harry_ccbft', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--input', required=True, type=str, help='Path to the input file.')
parser.add_argument('--output', required=True, type=str, help='Path to the output file.')
parser.add_argument('--no_local_salience', action="store_true", help='Whether to skip local salience prediction.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

def apply_fewshot_template_schema(examples):
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
    return template

def apply_inference_template_schema(example, previous_outputs=[]):
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

def build_fewshot_schema():
    with open("one_shot.pkl", "rb") as f:
        one_shot = pickle.load(f)

    selected_examples = [one_shot]
    fewshot = apply_fewshot_template_schema(selected_examples)
        
    return fewshot

def apply_fewshot_template_states(examples):
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
    return template

def build_fewshot_states():
    with open("one_shot.pkl", "rb") as f:
        one_shot = pickle.load(f)

    selected_examples = [one_shot]
    fewshot = apply_fewshot_template_states(selected_examples)
        
    return fewshot

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout))
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

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout))
def run_chatgpt(prompt, model="gpt-3.5-turbo", temperature=0.7):
    ret = openai.ChatCompletion.create(
        model=model,
        messages=prompt
        )
    gen_text = dict(ret["choices"][0]["message"])["content"]
    return gen_text

def predict_schema(example):
    prompt_fewshot = build_fewshot_schema()
    apply_template = apply_inference_template_schema
    run = run_gpt
    stop = ['\n']
    step_schema = []

    previous_outputs = []
    for _ in example["steps"]:
        prompt = prompt_fewshot + apply_template(example, previous_outputs)
        output = run(prompt, stop=stop)
        #print(output)
        # parse output
        output_str = output
        previous_outputs.append(output_str)
        pred_entities_attributes = {}

        for e_a in output_str.split('), '):
            try:
                entity = e_a.split(" (")[0]
                pred_entities_attributes[entity] = []
                for attribute in e_a.split(" (")[1].split(','):
                    processed_attribute = attribute.strip().strip('.').strip(')')
                    if processed_attribute:
                        pred_entities_attributes[entity].append(processed_attribute)
            except IndexError:
                continue
        
        step_schema.append(pred_entities_attributes)

    return step_schema

def parse_generated_text(gen_text):
    # Eraser: 5 - The eraser is the main component of the instruction and is essential for cleaning the inside of the windshield.
    #print(gen_text)
    try:
        score = re.search(r'\d+', gen_text).group()
        explanation = gen_text
    except:
        score = 1
        explanation = ""
        print("Error parsing generated text.", gen_text)
    return score, explanation

def predict_global_salience(goal, steps, entities):
    prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
               {"role": "user", "content": f"Here are some instructions on \"{goal}\".\n" + '\n'.join(['- ' + s for s in steps]) + "\n" + "Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance in the instruction. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
    prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
    output = []
    for entity in entities:
        prompt.append({"role": "user", "content": entity})
        gen_text = run_chatgpt(prompt)
        score, explanation = parse_generated_text(gen_text)
        output.append({"entity": entity, "global_salience_pred": int(score), "global_salience_explanation": explanation})
        prompt.append({"role": "assistant", "content": gen_text})
    return sorted(output, key=lambda d: d['global_salience_pred'], reverse=True) 

def predict_local_salience(goal, steps, entities):
    local_output = [[] for x in range(len(steps))]
    for i, step in enumerate(steps):
        prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
                {"role": "user", "content": f"One of the step of \"{goal}\" is \"{step}\". Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance for finishing this step. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
        prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
        for entity in entities:
            prompt.append({"role": "user", "content": entity})
            gen_text = run_chatgpt(prompt)
            score, explanation = parse_generated_text(gen_text)
            local_output[i].append({"entity": entity, "local_salience_pred": int(score), "local_salience_explanation": explanation})
            prompt.append({"role": "assistant", "content": gen_text})
        local_output[i] = sorted(local_output[i], key=lambda d: d['local_salience_pred'], reverse=True) 
    #print(local_output)
    return local_output

def predict_states(id, goal, steps, step_blocks):
    prompt_fewshot = build_fewshot_states()

    step_states = []
    prompt = prompt_fewshot + f"""A person's goal is to {goal.lower()}.
For each of the steps, list all the state changes of involved entities and attributes."""
    for i, step_block in enumerate(step_blocks):
        prompt += f"\nStep: {steps[i]}"
        step_pred = []
        for entity, attributes in step_block.items():
            for attribute in attributes:
                prompt += f"\n  - {attribute.split(' | ')[0]} of {entity.split(' | ')[0]} was"
                output = run_gpt(prompt, stop=['\n'])
                prompt += ' ' + output
                output_str = output
                pred_pre = output_str.strip().split(' before and ')[0]
                pred_post = output_str.strip().split(' before and ')[1].split(' after')[0]
                step_pred.append((entity, attribute, pred_pre, pred_post))
        
        step_states.append(step_pred)

    return step_states

if __name__ == "__main__":
    with open(args.input, "r") as f:
        examples = json.load(f)["input"]
    out_dict = {}
    for i, example in enumerate(examples):
        print(i)
        id = example["id"]
        out_dict[id] = {}
        goal = example["goal"]
        steps = example["steps"]
        step_schema = predict_schema(example)
        print(step_schema)
        out_dict[id]["schema"] = step_schema.copy()

        step_states = predict_states(id, goal, steps, step_schema)
        print(step_states)
        out_dict[id]["states"] = step_states.copy()

        all_entities = [[ent for ent in s.keys()] for s in step_schema]
        all_entities = [item for sublist in all_entities for item in sublist]
        all_entities = list(set(all_entities))
        print(all_entities)

        global_output = predict_global_salience(goal, steps, all_entities)
        print(global_output)
        out_dict[id]["global_salience"] = global_output.copy()
        if not args.no_local_salience:
            local_output = predict_local_salience(goal, steps, all_entities)
            #print(local_output)
            out_dict[id]["local_salience"] = local_output.copy()

    with open(args.output, "w") as f:
        json.dump(out_dict, f, indent=2)