import openai
import random
random.seed(299)
import json
import argparse
import backoff
import re

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt-3.5-turbo', type=str, help='Model name.')
parser.add_argument('--key', default='harry_ccbft', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--seed', default='', type=str, help='Random seed.')
parser.add_argument('--split', default='dev', type=str, help='The split to evaluate on.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()
if args.seed:
    random.seed(int(args.seed[1:]))

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def run_gpt(prompt, model=args.model, temperature=0.7):
    ret = openai.ChatCompletion.create(
        model=model,
        messages=prompt
        )
    gen_text = dict(ret["choices"][0]["message"])["content"]
    return gen_text

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

def predict_global(goal, steps, entities):
    prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
               {"role": "user", "content": f"Here are some instructions on \"{goal}\".\n" + '\n'.join(['- ' + s for s in steps]) + "\n" + "Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance in the instruction. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
    prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
    output = {}
    for entity in entities:
        prompt.append({"role": "user", "content": entity})
        gen_text = run_gpt(prompt)
        score, explanation = parse_generated_text(gen_text)
        output[entity] = {"global_salience_pred": score, "global_salience_explanation": explanation}
        prompt.append({"role": "assistant", "content": gen_text})
    return output

def predict_local(goal, steps, entities):
    local_output = [{} for x in range(len(steps))]
    for i, step in enumerate(steps):
        prompt= [{"role": "system", "content": "You will assign scores to objects in an intruction based on their importance"},
                {"role": "user", "content": f"One of the step of \"{goal}\" is \"{step}\". Now, I will provide you with a series of objects, and you will assign scores on a scale of 1-5 to them based on their importance for finishing this step. Your answer should strictly be a numerical score, followed by a one-sentence explanation."}]
        prompt.append({"role": "assistant", "content": "Sure, I can do that. Please provide me with the series of objects."})
        for entity in entities:
            prompt.append({"role": "user", "content": entity})
            gen_text = run_gpt(prompt)
            score, explanation = parse_generated_text(gen_text)
            local_output[i][entity] = {"local_salience_pred": score, "local_salience_explanation": explanation}
            prompt.append({"role": "assistant", "content": gen_text})
    #print(local_output)
    return local_output

with open("../data/dev-data-reformatted-v4.json") as f:
    data = json.load(f)
    for id, a in data.items():
        print(id)
        goal = a["goal"]
        steps = a["steps"]
        entities = [state["entity"] for state in a["states"]]
        global_output = predict_global(goal, steps, entities)
        local_output = predict_local(goal, steps, entities)
        for i, state in enumerate(a["states"]):
            #print(global_output[state["entity"]])
            data[id]["states"][i].update(global_output[state["entity"]])
            for j, step_num in enumerate(a["states"][i]["answers"]):
                data[id]["states"][i]["answers"][step_num] = {"attributes": data[id]["states"][i]["answers"][step_num]}
                data[id]["states"][i]["answers"][step_num].update(local_output[j][state["entity"]])
        if id == "20":
            break

with open("../data/dev-data-reformatted-v4_pred-salience.json", "w") as f_out:
    json.dump(data, f_out, indent=4)