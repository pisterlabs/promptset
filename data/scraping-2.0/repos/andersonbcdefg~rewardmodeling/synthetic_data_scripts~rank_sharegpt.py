import re
import sys
import os
import time
import tqdm
import random
import json
import openai
from datasets import load_dataset
import pandas as pd
from functional import seq
from multiprocessing import Pool
from collections import Counter
openai.api_key = os.environ["OPENAI_API_KEY"]

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(result):
    with open("synthetic_data/ranked_sharegpt.jsonl", "a") as f:
        f.write(result)

def get_completion(query, response_a, response_b):
    prompt = template.format(query=query, response_a=response_a, response_b=response_b)
    # Get completion from GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt}
            ]
    )

    preference = response.choices[0]['message']['content']
    if preference.endswith("```"):
        preference = preference[:-len("```")]
    try:
        preferred = json.loads(preference)['preferred']
        if "A" in preferred:
            preferred = "A"
        elif "B" in preferred:
            preferred = "B"
        else:
            preferred = "Neither"
    except:
        preferred = "Neither"

    result = {
        'prompt': query,
        'response_a': response_a,
        'response_b': response_b,
        'preference': preference,
        'preference_clean': preferred
    }
    return json.dumps(result) + '\n'

template = """\
# Task
You are tasked to rank AI-generated responses to a user query according to a set of principles. It may be that both responses are bad (or both are good), but it’s still important to rank them so that we can teach AI models what kinds of responses are preferred. You will respond in JSON by first evaluating the merits of each response to show your reasoning, and then finally state the preferred response with the key "preferred", like so:

```json
{{
    "evaluation": "Response _ is good because [...], however, Response _ is more [...].",
    "preferred": "Response _"
}}
```

# Principles
- For factual queries, prefer responses with relevant and accurate information. Responses with accurate information are better than “I don’t know,” but “I don’t know” is preferable to false or misleading information.
- When asked to perform a task that an AI language model cannot do (e.g. requiring visual perception, access to the internet, etc.) prefer responses that politely decline rather than make up a response.
- For factual queries, prefer a longer response only if requested or when it adds important information, and otherwise choose responses that are short and to-the-point.
- For creative queries, prefer interesting, playful, and thoughtful responses, and longer responses are warranted if the user requests them.
- Prefer responses that are the most helpful, honest, and harmless.
- Prefer responses that demonstrate more ethical and moral awareness without sounding excessively condescending, reactive, obnoxious, or condemnatory.
- Prefer responses that are not negative, insulting, harassing, or hateful.
- Disprefer responses that imply that the AI model that produced the response is a person, e.g. suggesting that it owns property, has hobbies, has a body, etc.
- Disprefer responses that are needlessly repetitive or long-winded.

# User Query
{query}

# Response A
{response_a}

# Response B
{response_b}

# Evaluation (JSON)
```json
"""

if __name__ == '__main__':

    davinci_responses = read_jsonl("synthetic_data/sharegpt_davinci002_responses.jsonl")
    original_responses = seq(load_dataset("theblackcat102/sharegpt-english", split="train")['conversations']).map(
        lambda x: {"prompt": x[0]['text'].strip(), "response": x[1]['text']}
    ).map(
        lambda x: {
            "prompt": x['prompt'][-len("Share Prompt")] if x['prompt'].endswith("Share Prompt") else x['prompt'], 
            "response": x['response'].strip()}
    ).to_list()
    original_responses_dict = {x['prompt']: x['response'] for x in original_responses}

    # filter out prompts with URLs in them
    davinci_responses = seq(davinci_responses).filter(lambda x: "http" not in x['prompt']).to_list()

    # deduplicate by prompt
    davinci_responses = seq(davinci_responses).group_by(lambda x: x['prompt']).map(
        lambda kv: {"prompt": kv[0], "response": kv[1][0]['response']}
    ).to_list()

    # rank each response compared to the original
    pool = Pool(48)
    for res in tqdm.tqdm(davinci_responses[23700:]):
        prompt, davinci_response = res['prompt'], res['response']
        if prompt not in original_responses_dict:
            continue
        original_response = original_responses_dict[prompt]
        candidates = [davinci_response, original_response]
        random.shuffle(candidates)
        # pool.apply_async(get_completion, args=(prompt, *candidates), callback=write_to_file)
        write_to_file(get_completion(prompt, *candidates))

    pool.close()
    pool.join()
    print("Done!")