import time
import openai
import json
from tqdm import tqdm
import numpy as np
from arguments import get_args
from reward import Reward
from utils.utils import batchify
from utils.constants import HOME_PATH
from utils.constants import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def get_gpt3_output(queries, model, temperature=1.0, max_tokens=128, top_p=0.9):
    attempts = 1
    while attempts <= 20:
        try:
            if model.startswith('text') or model.startswith('davinci:'):
                response = openai.Completion.create(
                    model=model,
                    prompt=queries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                generation = [c.text.strip() for c in response.choices]
            else:
                messages = [{"role": "user", "content": queries[0]}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                generation = [response.choices[0]['message']["content"].strip()]
            break
        except Exception as e:
            attempts += 1
            print(f"Service unavailable, retrying in 10 seconds ({attempts}/5): {e}")
            time.sleep(10)
    else:
        return None

    return generation


#model_name = 'gpt-3.5-turbo'
model_name = 'davinci:ft-mosaic-2023-05-14-21-09-15'
batch_size = 12 if model_name.startswith('text') else 1
datasets = json.load(open(f'{HOME_PATH}/data/commongen/test.json', 'r'))
args = get_args()
evaluator = Reward('output', 32, 0, args)

prompts, concepts = [], []
for key, item in datasets.items():
    for order_words in item['human_order'][:1]:
        prompt = 'Generate a sentence including the following keywords in the same order as listed: %s\n\nAnswer:'
        prompt = prompt % ' '.join(order_words.split('-'))
        prompts.append(prompt)
        concepts.append(order_words)

generations = []
for batch_prompts in tqdm(batchify(prompts, batch_size=batch_size), total=len(prompts) // batch_size):
    outputs = get_gpt3_output(batch_prompts, model_name, temperature=1.0, max_tokens=64, top_p=0.9)
    generations.extend(outputs)
generations = [g.split('.')[0] + '.' for g in generations]

score_dict = evaluator.get_reward(prompts, generations, concepts, f'eval_{model_name}')
for name, scores in score_dict.items():
    metric_score = np.mean(scores)
    print(f"  {name} = {metric_score:+.4f}")