"""
Analyze the distribution of a single 'randomly generated' digit from a LLM
"""

#TODO(ltang): write control flow for logit saving vs. sampling

import argparse
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import pickle
import random
import re
import torch
from api_keys import OPENAI_API_KEY
from collections import defaultdict
from scipy import stats
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaTokenizer, LlamaForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List
from watermark_playground import SingleLookbackWatermark


os.environ['KMP_DUPLICATE_LIB_OK']='True'
openai.api_key = OPENAI_API_KEY
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
CACHED_MODEL = None
CACHED_TOKENIZER = None

# TODO(ltang): clean this up later
td3_prompt = "Pick a random number between 1 and 100. Just return the number, don't include any other text or punctuation in the response."
gpt2_prompt = "What value would random.randint(1, 100) produce?"
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate a random number between 1 and 100.

### Response:"""
flan_prompt = "Pick a random integer between 1 and 100. Just return the number, don't include any other text or punctuation in the response."


def generate_from_model(
    model, 
    model_name,
    input_ids, 
    tokenizer,
    length: int,
    decode: str = 'beam',
    num_beams: int = 4,
    repetition_penalty: float = 1.0,
    logits_processors = [],
    rng: bool = True,
):

    if decode == 'beam':
        beam_count = num_beams
        do_sample = True
    elif decode == 'multinomial':
        beam_count = 1
        do_sample = True
    elif decode == 'greedy':
        beam_count = 1
        do_sample = False
    else:
        raise Exception

    if rng:
        sample = True
        scores = False
        return_dict = False
    else: 
        sample = False
        scores = True
        return_dict = True

    outputs = model.generate(
        input_ids,
        min_length=length,
        max_new_tokens=length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=logits_processors,
        do_sample=sample,
        output_scores=scores,
        return_dict_in_generate=return_dict,
    )

    if rng:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("Generated text", generated_text)

        # Postprocess response according to model type
        if model_name == 'openai-api':
            split_text = td3_prompt
        elif model_name == 'gpt-2':
            split_text = gpt2_prompt
        elif model_name == 'alpaca-lora':
            split_text = alpaca_prompt
        elif model_name.startswith('flan-t5'):
            split_text = flan_prompt
        
        try:
            response_text = generated_text.split(split_text)[-1]
        except Exception as e:
            print(f"Exception during postprocess: {e}")
        
        # Return first integer within acceptable range encountered in generation
        try:
            token = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", response_text)[0])
            if token in set(range(1, 101)):
                return int(token)
        except:
            # No valid numbers found
            return None
        
    else: 
        print("outputs.scores?", outputs.scores)
        return outputs.scores


def generate_random_digit(
    prompt, 
    tokenizer, 
    model_name='openai-api', 
    model=None, 
    engine='text-davinci-003', 
    length=10,
    watermark=None,
    decode='beam',
    rng=True,
):
    """
    - 'Randomly' generate a digit from a given range (not necessarily binary) or save logits
    - See prompt iterations in `prompt_list.txt`
    """

    assert decode in ['beam', 'greedy', 'multinomial']
    
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    processor = [watermark] if watermark else []
    repetition_penalty = 1

    if model_name == 'openai-api':
        # TODO(ltang): figure out how to extend logit bias in OpenAI model
        if watermark:
            logit_bias = {}
            # Vocab size esimate from https://enjoymachinelearning.com/blog/the-gpt-3-vocabulary-size/
            vocab_size = 175000
            green_list_length = int(vocab_size * watermark.gamma)
            # TODO(ltang): hash from previous tokens then seed generator here
            indices_to_mask = random.sample(range(vocab_size), green_list_length)
            # Note that logit_bias persists for the entire generation.
            # So potentially want to re-feed in prompt
            for idx in indices_to_mask:
                logit_bias[idx] = watermark.delta
        else:
            logit_bias = None
        while True:
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    # Though we want to generate a single number, there may be misc. characters like "\n" and such
                    # So allow for slightly larger `max_tokens`
                    max_tokens=length, 
                    temperature=1,
                    logprobs=2,
                    logit_bias=logit_bias,
                )

                raw_digit = response.choices[0].text.strip()
                int_digit = int(raw_digit)
            except Exception as e:
                # Catch when generated `raw_digit` is not of `int` type or general OpenAI server error 
                print(f"Exception: {e}")
                continue
            else:
                break
        
        return int_digit

    else:
        if rng:
            while True:
                int_digit = generate_from_model(model, model_name, input_ids, tokenizer, length, decode, logits_processors=processor, repetition_penalty=1, rng=rng)
                if int_digit is not None:
                    print("Returned int_digit that is not None", int_digit)
                    return int_digit
        else:
            return generate_from_model(model, model_name, input_ids, tokenizer, length, decode, logits_processors=processor, repetition_penalty=1, rng=rng)


def plot_digit_frequency(digits, output_file):

    print("Raw digits!", digits)
    digit_counts = {}
    
    # TODO(ltang): don't hardcode this range in the future
    for i in range(1, 101):
        digit_counts[i] = 0 

    for d in digits:
        if d in set(range(1, 101)): 
            digit_counts[d] += 1
        else:
            digit_counts[d] = 500

    now = datetime.datetime.now()
    file_label = f"{str(now.month)}-{str(now.day)}-{str(now.hour)}-{str(now.minute)}"
    numbered_out_file = output_file.split('.')[0] + '_' + file_label + '.json'
    with open(numbered_out_file, 'w') as file:
        json.dump(digit_counts, file, indent=4)
        
    plt.hist(digits, bins=100, range=(0, 100), alpha=0.7, density=True)
    plt.xlabel('Generated Number')
    plt.ylabel('Frequency')
    plt.title(f'RNG Frequencies for {len(digits)} Samples')
    plt.legend()
    png_file = numbered_out_file.split('.')[0] + '.png'
    plt.savefig('test.png')


def repeatedly_sample(prompt, model_name, engine="text-davinci-003", decode='beam', length=10, repetitions=2000, watermark=None, rng=True) -> List:

    assert model_name in ['openai-api', 'gpt-2', 'alpaca-lora', 'flan-t5']
    global CACHED_MODEL
    global CACHED_TOKENIZER

    if CACHED_MODEL is None or CACHED_TOKENIZER is None:
        if model_name == 'openai-api':
            tokenizer = None
            model = None
        elif model_name == 'gpt-2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
            vocab_size = model.config.vocab_size
        elif model_name == 'alpaca-lora':
            tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
            print("Done Loading Alpaca Tokenizer")
            model = LlamaForCausalLM.from_pretrained(
                "chainyo/alpaca-lora-7b",
                torch_dtype=torch.float16
            ).to(device)
            print("Done Loading Alpaca Model")
        elif model_name.startswith("flan-t5"):
            tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
            print("Done Loading Flan Tokenizer")
            model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xxl').to(device)
            print("Done Loading Flan Model")

        CACHED_TOKENIZER = tokenizer
        CACHED_MODEL = model

    else:
        model = CACHED_MODEL
        tokenizer = CACHED_TOKENIZER
    
    if rng:
        print(f"Sampling for {repetitions} repetitions")
        sampled_digits = []
        for i in range(repetitions):
            if i % (repetitions // 5) == 0: print(f'On repetition {i} of {repetitions}')
            d = generate_random_digit(prompt, tokenizer, model_name, model=model, length=length, decode=decode, engine=engine, watermark=watermark, rng=rng)
            sampled_digits.append(d)
        
        return sampled_digits

    else:
        logits_tuple = generate_random_digit(prompt, tokenizer, model_name, model=model, length=length, decode=decode, engine=engine, watermark=watermark, rng=rng)
        logits_tuple = [l.cpu() for l in logits_tuple]
        return logits_tuple



def KL(P, Q):
    """ 
    - Calculate KL divergence between P and Q
    - Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0 
    """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


def KL_loop(prompt, model, length, num_dists, out_file, gamma, delta, kl=False, rng=True):
    """
    - Generate `num_dists` distributions and compute pairwise KL between them.
    - Generate violin plot
    - Return full list of KLs and also their summary stats
    """

    distributions = []
    pairwise_KLs = []

    # Kirchenbauer watermark
    if gamma is not None and gamma > 0 and delta is not None and delta > 0:
        watermark = SingleLookbackWatermark(gamma=gamma, delta=delta)
    else: 
        watermark = None

    assert not (kl and not rng)

    if rng:
        for _ in range(num_dists):
            if model == 'openai-api':
                digit_sample = repeatedly_sample(prompt, 'openai-api', engine='text-davinci-003', decode='beam', length=10, repetitions=1000, watermark=watermark, rng=rng)
            elif model == 'alpaca-lora':
                digit_sample = repeatedly_sample(prompt, 'alpaca-lora', decode='beam', length=length, repetitions=1000, watermark=watermark, rng=rng)
            elif model == 'flan-t5':
                digit_sample = repeatedly_sample(prompt, 'flan-t5', decode='beam', length=length, repetitions=1000, watermark=watermark, rng=rng)
            
            # TODO(ltang): We can probably also take a KL between each massive distribution (just sum up across each of 1000 dists)
            distributions.append(np.array(digit_sample))

        raw_data = np.array(distributions)
        now = datetime.datetime.now()
        file_label = f"{str(now.month)}-{str(now.day)}-{str(now.hour)}-{str(now.minute)}"
        numbered_out_file = out_file.split('.')[0] + '_' + file_label + '.npy'
        np.save(numbered_out_file, raw_data)

        if kl:
            for i, d_1 in enumerate(distributions):
                for j, d_2 in enumerate(distributions):
                    # TODO(ltang): think about alternative (ideally symmetric) divergence metric
                    if i >= j: continue
                    kl = KL(d_1, d_2)
                    pairwise_KLs.append(kl)

            kl_data = np.array(pairwise_KLs)
            fig, ax = plt.subplots()
            ax.violinplot(kl_data, showmeans=False, showmedians=True)
            ax.set_title('Distribution of Pairwise KLs for Unmarked Model')
            ax.set_ylabel('KL Divergence')

            png_file = numbered_out_file.split('.')[0] + '.png'
            plt.savefig(png_file)

            print("Full Pairwise KL List:", pairwise_KLs)
            print("Pairwise KL Summary Statistics")
            print(stats.describe(kl_data))
    
    # Generate logits instead
    else:
        if model == 'openai-api':
            raise Exception('OpenAI Models do not return logits')
        elif model == 'alpaca-lora':
            return repeatedly_sample(prompt, 'alpaca-lora', decode='beam', length=length, repetitions=1000, watermark=watermark, rng=rng)
        elif model == 'flan-t5':
            return repeatedly_sample(prompt, 'flan-t5', decode='beam', length=length, repetitions=1000, watermark=watermark, rng=rng)


def plot_example():
    watermark = SingleLookbackWatermark(gamma=0.5, delta=10)
    digit_sample = repeatedly_sample(prompt, 'openai-api', engine='text-davinci-003', decode='beam', length=10, repetitions=1000, watermark=watermark)
    plot_digit_frequency(digit_sample, 'misc/digit_counts_td3_05_10.json')

    ## Misc. Examples
    # KL_loop(10, 'td3_unmarked_rep_10.npy', 0.5, 10)


def main(model, prompt, rng=True, save_version=''):
    """
    Meta-loop over watermark parameters and generate random number distribution at each parameter setting
    """

    for gamma in [0, 0.1, 0.25, 0.5, 0.75]:
        for delta in [0, 1, 5, 10, 50, 100]:
            print(f"KL Loop for gamma {int(gamma * 100)} and delta {delta}")
            
            # Unmarked model
            if gamma == 0 and delta == 0:
                if rng:
                    if model == 'flan-t5':
                        KL_loop(prompt, model, 10, 10, f'flan_unmarked.npy', None, None)
                    if model == 'alpaca-lora':
                        KL_loop(prompt, model, 10, 10, f'alpaca_unmarked.npy', None, None)
                else:
                    if model == 'flan-t5':
                        logits_tuple = KL_loop(prompt, model, 10, 10, f'flan_marked_g{int(gamma * 100)}_d{delta}_rep_10.npy', gamma, delta, rng=False)
                        with open(f'flan_logits_unmarked{save_version}.pt', 'wb') as f:
                            pickle.dump(logits_tuple, f)
                    if model == 'alpaca-lora':
                        logits_tuple = KL_loop(prompt, model, 10, 10, f'alpaca_marked_g{int(gamma * 100)}_d{delta}_rep_10.npy', gamma, delta, rng=False)
                        with open(f'alpaca_logits_unmarked{save_version}.pt', 'wb') as f:
                            pickle.dump(logits_tuple, f)

            # Can skip these conditions since we have already processed unmarked model
            elif gamma == 0 or delta == 0:
                continue
            
            # Watermarked model of varying strengths
            else:
                # Generate random number distributions
                if rng:
                    if model == 'flan-t5':
                        KL_loop(prompt, model, 10, 10, f'alpaca_marked_g{int(gamma * 100)}_d{delta}_rep_10.npy', gamma, delta)
                    if model == 'alpaca-lora':
                        KL_loop(prompt, model, 10, 10, f'flan_marked_g{int(gamma * 100)}_d{delta}_rep_10.npy', gamma, delta)
                # Save logits instead of generating random number distribution
                else:
                    if model == 'flan-t5':
                        logits_tuple = KL_loop(prompt, model, 10, 10, f'flan_marked_g{int(gamma * 100)}_d{delta}_rep_10.npy', gamma, delta, rng=False)
                        with open("flan_logits_marked_g{}_d{}{}.pt".format(int(gamma * 100), delta, save_version), 'wb') as f:
                            pickle.dump(logits_tuple, f)
                    if model == 'alpaca-lora':
                        logits_tuple = KL_loop(prompt, model, 10, 10, f'alpaca_marked_g{int(gamma * 100)}_d{delta}_rep_10.npy', gamma, delta, rng=False)
                        with open("alpaca_logits_marked_g{}_d{}{}.pt".format(int(gamma * 100), delta, save_version), 'wb') as f:
                            pickle.dump(logits_tuple, f)
                    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--logits', '-l', action='store_true')
    args = parser.parse_args()

    with open('openwebtext.json', 'r') as f:
        pile_dict = json.load(f)

    # print("pile_dict", pile_dict)
    texts = []

    def return_texts(d, output):
        for key, value in d.items():
            # print("key", key)
            if key == 'text':
                output.append(value)
            if isinstance(value, dict):
                return_texts(value, output)
            if isinstance(value, list):
                for row in value:
                    return_texts(row, output)

    return_texts(pile_dict, texts)

    prompts = [t + ". Now write me a story:" for t in texts]

    for i, p in enumerate(prompts):
        main(args.model, p, rng=(not args.logits), save_version=f'_v{i + 51}')
