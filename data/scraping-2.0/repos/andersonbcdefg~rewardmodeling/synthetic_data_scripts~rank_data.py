import os
import time
import tqdm
import json
import openai
import pandas as pd
from multiprocessing import Pool
openai.api_key = os.environ["OPENAI_API_KEY"]

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(result):
    with open("ranked_responses3.jsonl", "a") as f:
        f.write(result)

def get_completion(query, response_a, response_b):
    prompt = template.format(query=query, response_a=response_a, response_b=response_b)
    # Get completion from GPT-4
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "user", "content": prompt}
        ]
    )

    preference = response.choices[0]['message']['content']
    result = {
        'prompt': query,
        'response_a': response_a,
        'response_b': response_b,
        'preference': preference
    }
    return json.dumps(result) + '\n'

template = """\
# Task
You are tasked to rank AI-generated responses to a user query according to a set of principles. It may be that both responses are bad, or both are good, but it’s still important to rank them so that we can teach AI models what kinds of responses are preferred. Respond by first evaluating the merits of each response to show your reasoning, and then finally state the preferred response on a new line, like so:

Preferred response: Response B
or
Preferred response: Response A

You MUST pick, even if both responses are bad. If you cannot decide, then pick the response that is the least bad.

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

# Evaluation\
"""

if __name__ == '__main__':

    base_dir = '/Users/ben/Downloads/synthetic_dolly/'

    combined_df = None
    for i in range(1, 7):
        results = read_jsonl(base_dir + f'results{i}.jsonl')
        df = pd.DataFrame.from_records(results)
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df])

    print(len(combined_df))
    grouped_df = combined_df.groupby('prompt')['response'].agg(list).reset_index()
    grouped_df.columns = ['prompt', 'responses']

    # Convert the grouped DataFrame into a dictionary
    prompt_responses_dict = grouped_df.set_index('prompt')['responses'].to_dict()

    # Rank two responses for each prompt
    results = []
    pool = Pool(8)

    # filter out prompts that have already been ranked
    existing_ranks = read_jsonl("ranked_responses3.jsonl")
    existing_ranks = [r['prompt'] for r in existing_ranks if 'Preferred response: Response' in r['preference']]
    prompt_responses_dict = {k: v for k, v in prompt_responses_dict.items() if k not in existing_ranks}
    print("Number of prompts to rank:", len(prompt_responses_dict))
    for query, responses in tqdm.tqdm(list(prompt_responses_dict.items())):
        responses = list(set(responses))
        responses = [r.split("###")[0].strip() for r in responses]
        response_a = sorted(responses, key=lambda x: len(x))[-1]
        response_b = sorted(responses, key=lambda x: len(x))[-2]
        pool.apply_async(get_completion, args=(query, response_a, response_b), callback=write_to_file)
        time.sleep(0.1)

    pool.close()
    pool.join()
    print("Done!")