import re
import sys
import os
import time
import tqdm
import json
import openai
from datasets import load_dataset
import pandas as pd
from functional import seq
from multiprocessing import Pool
openai.api_key = os.environ["OPENAI_API_KEY"]

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(result):
    with open("dolly_ai_vs_human.jsonl", "a") as f:
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

    preferred_ai_responses = pd.DataFrame.from_records(read_jsonl("ranked_responses3.jsonl"))

    # use regex to extract Preferred response: Response A or B, if neither then "Neither"
    preferred_ai_responses['preference_clean'] = preferred_ai_responses.apply(
        lambda row: re.findall(r"Preferred response: Response (A|B)", row['preference'])[0] if len(re.findall(r"Preferred response: Response (A|B)", row['preference'])) > 0 else "Neither",
        axis=1
    )
    preferred_ai_responses = preferred_ai_responses[preferred_ai_responses['preference_clean'] != "Neither"]
    preferred_ai_responses['preferred_text'] = preferred_ai_responses.apply(
        lambda row: row['response_a'] if row['preference_clean'] == "A" else row['response_b'],
        axis=1
    )
    # remove ones with blank responses
    preferred_ai_responses = preferred_ai_responses[preferred_ai_responses['response_a'] != ""]
    preferred_ai_responses = preferred_ai_responses[preferred_ai_responses['response_b'] != ""]

    # dedupe by prompt
    preferred_ai_responses.groupby('prompt').first().reset_index()
    
    original_dolly = load_dataset("databricks/databricks-dolly-15k", split="train").filter(
        lambda example: example['context'] == ""
    ).to_pandas()
    original_dolly.columns = ['prompt', 'input', 'human_response', 'category']

    # Join with the original dolly responses
    combined = preferred_ai_responses.merge(original_dolly, on='prompt', how='left')
    print(combined.head()[['prompt', 'preferred_text', 'human_response']])

    # Rank preferred AI responses vs. Dolly for each prompt. Dolly is always response A.
    results = []
    pool = Pool(12)

    # zip the prompts, human response, and preferred AI response
    for query, response_a, response_b in list(zip(combined['prompt'], combined['human_response'], combined['preferred_text']))[10000:]:
        pool.apply_async(get_completion, args=(query, response_a, response_b), callback=write_to_file)
        time.sleep(0.1)

    pool.close()
    pool.join()
    print("Done!")