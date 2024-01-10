import time
from pathlib import Path
from tqdm import tqdm

import openai
from transformers import GPT2Tokenizer

from constants import *

API_KEYS_FILE = "keys.txt"
PREDICTIONS_DIR = "./predictions/"

class OpernAIParams:
    model="code-davinci-002"
    max_generation_tokens=128
    temperature=0
    top_p=1
    n=1
    top_p=1
    stop='\n'
    presence_penalty=0
    best_of=10

def read_dataset(filename):
    # Open source
    with open(Path(DATASET_DIR, filename), 'r') as f:
        dataset = []
        for sentence in f:
            dataset.append([int(x) for x in sentence.split(' ')[:-1]])

    return dataset


def generate_few_shot_prompt(tokenizer, source_train_dataset, target_train_dataset, n_few_shot=10):

    # Generate the few-shot prompt
    few_shot_prompt = "Translate the following sentence from English to French \n\n"

    for i in range(n_few_shot):
        source = source_train_dataset[i]
        source = [token for token in source if (token != tokenizer.bos_token_id and token != tokenizer.eos_token_id)]
        source = tokenizer.decode(source)
        few_shot_prompt += f"English: {source} \n"

        target = target_train_dataset[i]
        target = [token for token in target if (token != tokenizer.bos_token_id and token != tokenizer.eos_token_id)]
        target = tokenizer.decode(target)
        few_shot_prompt += f"French: {target} \n\n"

    return few_shot_prompt


def generate_last_shot_prompt(tokenizer, few_shot_prompt, source):

    source = [token for token in source if (token != tokenizer.bos_token_id and token != tokenizer.eos_token_id)]
    source = tokenizer.decode(source)

    prompt = few_shot_prompt + f"English: {source} \n"
    prompt += "French: "

    return prompt


def read_api_keys(api_keys_file):
    with open(api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    return keys


def make_query(keys, prompt, params):

    result = None
    key_index = 0

    start_time = time.time()
    while result is None:
        try:
            key_index = (key_index + 1) % len(keys)
            result = openai.Completion.create(
                api_key=keys[key_index],
                prompt=prompt,
                model=params.model,
                max_tokens=params.max_generation_tokens,
                temperature=params.temperature,
                n=params.n,
                top_p=params.top_p,
                stop=params.stop,
                presence_penalty=params.presence_penalty,
                best_of=params.best_of
            )
        except Exception as e:
            print(e, "Retry with key index: ", key_index)
            time.sleep(5)

    elapsed_time = time.time() - start_time
    return result, elapsed_time


def main():

    # Load datasets
    source_train_dataset = read_dataset('wmt14_en_test.src')
    target_train_dataset = read_dataset('wmt14_fr_test.trg')
    source_test_dataset = read_dataset('wmt14_en_test.src')
    # target_test_dataset = read_dataset('wmt14_fr_test.trg')

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer')

    # Generate few-shot prompt
    few_shot_prompt = generate_few_shot_prompt(tokenizer, source_train_dataset, target_train_dataset, n_few_shot=10)

    # Generate last-shot prompt and query
    api_keys = read_api_keys(api_keys_file=API_KEYS_FILE)
    open_ai_params = OpernAIParams()
    predicted_targets = []
    for source in tqdm(source_test_dataset):
        prompt = generate_last_shot_prompt(tokenizer, few_shot_prompt, source)
        result, _ = make_query(api_keys, prompt, open_ai_params)
        predicted_target = result['choices'][0]['text']
        predicted_targets.append(predicted_target)

    # Save results
    with open(Path(PREDICTIONS_DIR, 'wmt14_en_fr_llm.txt'), 'w') as f:
        for predicted_target in predicted_targets:
            f.write(predicted_target + '\n')



if __name__ == "__main__":
    main()