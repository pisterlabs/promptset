import os
import torch
import openai
import tiktoken
from tenacity import retry, wait_fixed
from rich.progress import track
import time
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")


def binary_search(x, low=-0.5, high=0, eps=1e-3):
    d = torch.zeros_like(x)
    # greedy call
    idx = x.argmax()
    d[idx] = 1

    num_calls = 1
    # double low if it's not low enough
    while (x + d * low).argmax() == idx:
        low *= 2
        num_calls += 1

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        # call to greedy
        if (x + d * mid).argmax() == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
        #print(low, high)
    return mid, idx, num_calls

def estimate_topk_logits(logits, K):
    """
    Estimate the diffs between the elements of the logits vector
    """

    # approximate logit diff of top word vs 2nd highest
    mask_vec = torch.zeros_like(logits)
    diffs = []
    idxs = []
    total_calls = 0
    for n in range(K):
        logit_diff, idx, num_calls = binary_search(logits - mask_vec * 100)
        mask_vec[idx] = 1
        diffs.append(logit_diff)
        idxs.append(idx)
        total_calls += num_calls

    estimated_logits = torch.tensor(diffs[:K]).cumsum(0)
    out = torch.full_like(logits, float("-inf"))
    out[idxs] = estimated_logits
    return out


def openai_api_calculate_cost(usage,model="gpt-3.5-turbo"):
    pricing = {
        'gpt-3.5-turbo': {
            'prompt': 0.0015,
            'completion': 0.002,
        },
        'gpt-3.5-turbo-16k': {
            'prompt': 0.003,
            'completion': 0.004,
        },
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06,
        },
        'gpt-4-16k': {
            'prompt': 0.06,
            'completion': 0.12,
        },
        'text-embedding-ada-002-v2': {
            'prompt': 0.0001,
            'completion': 0.0001,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage['prompt_tokens'] * model_pricing['prompt'] / 1000
    completion_cost = usage['completion_tokens'] * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    print(f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['total_tokens']:,} tokens")
    print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost

#@retry(wait=wait_fixed(1))
#@retry
def complete(
    message,
    logit_bias=dict(),
    model="gpt-3.5-turbo",
    system=None,
    verbose=False,
    temperature = 0,
    n = 1,
):
    system = "You are a helpful assistant." if system is None else system
    enc = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo-instruct":
        response = openai.Completion.create(
            model=model,
            prompt=message,
            temperature=temperature,
            max_tokens=1,
            logit_bias=logit_bias,
            n=n,
        )
        output = response.choices[0].text
        eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            max_tokens=1,
            logit_bias=logit_bias,
            n=n,
        )
        output = response.choices[0].message["content"]
        eos_idx = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>", "<|im_start|>"})[0]

    if response.choices[0].finish_reason == "length":
        idx = enc.encode(output)[0]
    elif response.choices[0].finish_reason == "stop":
        idx = eos_idx
    else:
        import pdb; pdb.set_trace()
    if verbose:
        openai_api_calculate_cost(response.usage, model)
    return idx, response

def binary_search_openai(prefix, logit_bias, model, low=-0.5, high=0, eps=1e-3, system=None, verbose=False):
    logit_bias = logit_bias.copy()
    # greedy call
    idx, _ = complete(prefix, logit_bias, model, system=system, verbose=verbose)
    logit_bias[idx] = low

    num_calls = 1
    # double low if it's not low enough
    while complete(prefix, logit_bias, model)[0] == idx:
        low *= 2
        logit_bias[idx] = low
        num_calls += 1

    # improve estimate
    mid = (high + low) / 2
    while high > low + eps:
        # call to greedy
        logit_bias[idx] = mid
        if complete(prefix, logit_bias, model)[0] == idx:
            high = mid
        else:
            low = mid
        mid = (high + low) / 2
        num_calls += 1
    return mid, idx, num_calls


def estimate_topk_logits_openai(prefix, model, K, system=None, verbose=False):
    """
    Estimate the diffs between the elements of the logits vector
    """
    enc = tiktoken.encoding_for_model(model)
    vocab_size = enc.n_vocab
    logit_bias = dict()
    diffs = []
    idxs = []
    total_calls = 0
    for n in track(range(K)):
        start = time.time()
        logit_diff, idx, num_calls = binary_search_openai(prefix, logit_bias, model, system=system, verbose=verbose)
        if verbose:
            print("binary search took", time.time() - start)
        logit_bias[idx] = -100
        diffs.append(logit_diff)
        idxs.append(idx)
        total_calls += num_calls
        if verbose:
            print("call per estimate", num_calls)

    estimated_logits = torch.tensor(diffs[:K]).cumsum(0)
    out = torch.full((vocab_size,), float("-inf"))
    out[idxs] = estimated_logits
    return out


def test_logit_bias():
    model = "gpt-3.5-turbo"
    prefix = "The Hamiltonian Monte Carlo algorithm (originally known as hybrid Monte Carlo) is a Markov chain Monte Carlo method for obtaining a sequence of random samples which converge to being distributed according to a target probability distribution for which direct sampling is difficult. This sequence can be used to estimate integrals with respect to the target distribution (expected values)."
    idxs = []
    for i in tqdm(range(10000)):
        logit_bias = {x: -100 for x in idxs}
        idx, _ = complete(prefix, logit_bias=logit_bias, model=model)
        if idx in idxs:
            print(idxs)
            print(idx, "repeated")
            break
        else:
            idxs.append(idx)
    _, responses = complete(prefix, logit_bias, model=model, n=128, temperature=1)
    enc = tiktoken.encoding_for_model(model)

    for choice in responses.choices:
        if choice.finish_reason != "stop":
            output = choice.message.content
            idx = enc.encode(output)[0]
            already_seen.append(idx in idxs)
        else:
            # already seen EOS
            already_seen.append(True)
    print("already seen %:", sum(already_seen) / len(already_seen))

def test_many_logit_bias():
    import random
    model = "gpt-3.5-turbo"
    prefix = "The Hamiltonian Monte Carlo algorithm (originally known as hybrid Monte Carlo) is a Markov chain Monte Carlo method for obtaining a sequence of random samples which converge to being distributed according to a target probability distribution for which direct sampling is difficult. This sequence can be used to estimate integrals with respect to the target distribution (expected values)."
    enc = tiktoken.encoding_for_model(model)

    idxs = {random.randint(0, 100000) for n in range(8000)}
    logit_bias = {x: -100 for x in idxs}

    _, responses = complete(prefix, logit_bias, model=model, n=128, temperature=1)
    already_seen = []
    for choice in responses.choices:
        if choice.finish_reason != "stop":
            output = choice.message.content
            idx = enc.encode(output)[0]
            already_seen.append(idx in idxs)
        else:
            # already seen EOS
            already_seen.append(True)
    print("already seen %:", sum(already_seen) / len(already_seen))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    #test_logit_bias()
    #test_many_logit_bias()
    
    #model = "gpt-3.5-turbo-instruct"
    model = "gpt-3.5-turbo"
    prefix = "hi"
    prefix = "The Hamiltonian Monte Carlo algorithm (originally known as hybrid Monte Carlo) is a Markov chain Monte Carlo method for obtaining a sequence of random samples which converge to being distributed according to a target probability distribution for which direct sampling is difficult. This sequence can be used to estimate integrals with respect to the target distribution (expected values)."
    idx, _ = complete(prefix, model=model)

    #logit_bias = dict()
    output = binary_search_openai(prefix, logit_bias, model)
    logits = estimate_topk_logits_openai(prefix, model, 25, verbose=True)

    if model == "gpt-3.5-turbo-instruct":
        response = openai.Completion.create(
            model=model,
            prompt=prefix,
            temperature=0,
            max_tokens=1,
            logit_bias=dict(),
            logprobs=5,
        )
    import pdb; pdb.set_trace()
