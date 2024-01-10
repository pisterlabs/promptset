"""
This file isn't used in our final version.
"""
import sys
import fire
import json
import logging
import regex as re
import copy
import random
sys.path.append("..")
from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
from typing import List, Dict
from string import Template
from collections import Counter, defaultdict
logging.basicConfig(level=logging.WARNING)

template = """
${instruction}
${source}

A correct output is: 
${reference}

A model generated output is:
${model1_generated}

Now please evaluate the errors in the model-generated outputs
For each error associated with problem understanding, problem formulation,  computing accuracy, and solution interpretation, reduce 1 or 2 score. 
Finally give me a total reductions of score as the evaluation of this model-generated output starting with "Total Score Reduction: ".
"""


def get_prompts(
    item: dict
):
    prompts = []
    random.shuffle(item['candidates'])
    for cand in item['candidates']:
        prompt = Template(template).substitute(
            instruction=item['instruction'].strip("\n "),
            source=item['input'].strip("\n "),
            reference=(item.get('output') or item.get("refs")[0]).strip("\n "),
            model1_generated=cand['text'].strip("\n "),
        )
        prompts.append(prompt)
    return prompts

def main(
    input_file: str,
    output_file: str,
    seed: int = 42,
    model_name: str = "ChatGPT",
):
    random.seed(seed)
    with open(input_file, "r") as f:
        data = json.load(f)
    
    prompts = list(map(get_prompts, data))
    flatten_prompts = [prompt for prompts_ in prompts for prompt in prompts_]
    chatmls = [[{"role":"system","content":"You are an helpful AI assistant to help user find information."},
            {"role":"user","content": prompt}] for prompt in flatten_prompts]
    chatml_prompts = [_chatml_to_prompt(chatml) for chatml in chatmls]

    decoding_kwargs = {
        # "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "timeout": 30,
        "request_timeout": 30
    }
    results = openai_completions(chatml_prompts, model_name=model_name, **decoding_kwargs)
    logging.warning("Total price: {:.4f}$".format(sum(results['price_per_example'])))
    completions = results['completions']
    
    idx = 0
    for i, item in enumerate(data):
        for j, cand in enumerate(item['candidates']):
            total_score_reduction = re.search("Total Score Reduction: (\d+)", completions[idx])
            if not total_score_reduction:
                total_score_reduction = re.search("Total Score Reduction: -(\d+)", completions[idx])
            if not total_score_reduction:
                total_score_reduction = re.search("Total Score Reduction is (\d+)", completions[idx])
            if not total_score_reduction:
                total_score_reduction = re.search("Total Score Reduction is -(\d+)", completions[idx])
            if total_score_reduction:
                cand['scores']['gpt_score_reduction'] = - abs(int(total_score_reduction.groups()[0]))
            else:
                pass
                cand['scores']['gpt_score_reduction'] = 0
            cand['gpt_score_output'] = completions[idx]
            idx += 1
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.warning(f"Saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)