import openai
import os
import time
from copy import deepcopy
from tqdm import trange
import numpy as np

def gpt3wrapper(max_repeat=20, **arguments):
    openai.api_key = os.environ['openai_key']
    i = 0
    while i < max_repeat:
        try:
            start_time = time.time()
            response = openai.Completion.create(**arguments)
            end_time = time.time()
            # print('completed one query in', end_time - start_time)
            return response
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(arguments['prompt'])
            print(e)
            print('now sleeping')
            time.sleep(30)
            i += 1
    return None


def gpt3wrapper_texts(max_repeat=20, **arguments):
    openai.api_key = os.environ['openai_key']
    response = gpt3wrapper(max_repeat=max_repeat, **arguments)
    if response is None:
        return None
    if type(arguments['prompt']) == list:
        return [r['text'] for r in response['choices']]
    else:
        return response['choices'][0]['text']


eps = 1e-5
def extract_probs(response):
    yes_probs = []
    for r in response['choices']:
        d = r['logprobs']['top_logprobs'][0]
        yes_prob, no_prob = eps, eps
        for k, v in d.items():
            if 'yes' in k.lower():
                yes_prob += np.e ** v
            elif 'no' in k.lower():
                no_prob += np.e ** v
        yes_prob = yes_prob / (yes_prob + no_prob)
        yes_probs.append(yes_prob)
    return yes_probs


def gpt3wrapper_yes_no(max_repeat=20, **arguments):
    arg_copy = deepcopy(arguments)
    openai.api_key = os.environ['openai_key']
    arg_copy['logprobs'], arg_copy['max_tokens'] = 5, 1
    response = gpt3wrapper(max_repeat=max_repeat, **arg_copy)
    if response is None:
        return None
    if type(arguments['prompt']) == list:
        return extract_probs(response)
    else:
        return extract_probs(response)[0]

def gpt3wrapper_texts_batch(max_repeat=20, bsize=20, verbose=False, **arguments):
    openai.api_key = os.environ['openai_key']
    prompt = arguments['prompt']

    if type(prompt) == list:
        return list(gpt3wrapper_texts_batch_iter(max_repeat=max_repeat, bsize=bsize, verbose=verbose, **arguments))
    else:
        assert type(prompt) == str
        return gpt3wrapper_texts(max_repeat=max_repeat, **arguments)


def gpt3wrapper_texts_batch_iter(max_repeat=20, bsize=20, verbose=False, **arguments):
    openai.api_key = os.environ['openai_key']
    prompt = arguments['prompt']

    assert type(prompt) == list
    num_batches = (len(prompt) - 1) // bsize + 1
    iterator = trange(num_batches) if verbose else range(num_batches)
    for i in iterator:
        arg_copy = deepcopy(arguments)
        arg_copy['prompt'] = prompt[i*bsize:(i+1)*bsize]
        response = gpt3wrapper(max_repeat=max_repeat, **arg_copy)
        if response is None:
            for _ in range(len(arg_copy['prompt'])):
                yield None
        else:
            for text in [r['text'] for r in response['choices']]:
                yield text

def gpt3wrapper_yes_no_batch(max_repeat=20, bsize=20, verbose=False, **arguments):
    openai.api_key = os.environ['openai_key']
    prompt = arguments['prompt']

    if type(prompt) == list:
        num_batches = (len(prompt) - 1) // bsize + 1
        yes_no_probs = []
        iterator = trange(num_batches) if verbose else range(num_batches)
        for i in iterator:
            arg_copy = deepcopy(arguments)
            arg_copy['prompt'] = prompt[i*bsize:(i+1)*bsize]
            arg_copy['logprobs'], arg_copy['max_tokens'] = 5, 1
            response = gpt3wrapper(max_repeat=max_repeat, **arg_copy)
            if response is None:
                yes_no_probs.extend([None] * len(arg_copy['prompt']))
            else:
                yes_no_probs.extend(extract_probs(response))
        assert len(yes_no_probs) == len(prompt)
        return yes_no_probs
    else:
        return gpt3wrapper_yes_no(max_repeat=max_repeat, **arguments)

if __name__ == '__main__':
    # test
    prompts = [f'Calculate the result.\n1 + 1 = 2\n1 + {n}' for n in range(30)]
    responses = gpt3wrapper_texts_batch(prompt=prompts, max_tokens=3, temperature=0.0, max_repeat=20, engine='text-davinci-003')
    print(responses)

    prompts = [
        f'I like sushi.\nIs the above statement positive? answer yes or no.',
        f'I hate sushi.\nIs the above statement positive? answer yes or no.', 
    ] * 30
    responses = gpt3wrapper_yes_no_batch(prompt=prompts, max_tokens=3, temperature=0.0, max_repeat=20, engine='text-davinci-003')
    print(responses)