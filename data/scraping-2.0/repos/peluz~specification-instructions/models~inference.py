import json
import requests
import more_itertools as mit
import math
import openai
from tqdm.auto import tqdm
import random
import time
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download



def openai_request(prompts, api_token, format_instruction,  temperature=0.0, max_tokens=20):
    openai.api_key = api_token
    responses = []
    requests = [[{"role": "user", "content": prompt + f"{format_instruction}"}] for prompt in prompts]
    for request in tqdm(requests):
        while True:
            num_retries = 0
            delay = 1.
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=request,
                max_tokens=max_tokens,
                temperature=0,
                )
                responses.append(response)
                break
            except Exception as e:
                num_retries += 1
                print(e)
 
                # Check if max retries has been reached
                if num_retries > 10:
                    raise Exception(
                        f"Maximum number of retries (10) exceeded."
                    )
 
                # Increment the delay
                delay *= 2 * (1 + random.random())
                print(f"Retrying with delay {delay}")
 
                # Sleep for the delay
                time.sleep(delay)
    all_preds = [response["choices"][0]["message"]["content"] for response in responses]
    return responses, all_preds

def huggingface_request(prompts, model, api_token, prompts_per_req=128, temperature=0.0):
    all_preds = []
    batches = mit.batched(prompts, prompts_per_req)
    n_reqs = math.ceil(len(prompts)/prompts_per_req)
    headers = {"Authorization": f"Bearer {api_token}"}
    model_url = f"https://api-inference.huggingface.co/models/{model}"
    def query(**kwargs):
        data = json.dumps(kwargs)
        delay = 1.
        while True:
            response = requests.request("POST", model_url, headers=headers, data=data)
            result =  json.loads(response.content.decode("utf-8"))
            if response.status_code != 200:
                delay *= (1 + random.random())
                print(result)
                print(f"Retrying with delay {delay}")
                print()
                time.sleep(delay)
            else:
                return result
    parameters = {
        "temperature": temperature
    }
    for prompts in tqdm(batches, total=n_reqs):
        all_preds.extend(query(inputs=prompts, parameters=parameters))
    return all_preds

def load_model(model_url, num_gpus=1):
    if num_gpus > 8:
        device_map  = "balanced_low_0"
    else:
        device_map = "auto"
    if num_gpus > 1:
        with init_empty_weights():
            config = AutoConfig.from_pretrained(model_url)
            model = AutoModelForSeq2SeqLM.from_config(config)
        weights_location = snapshot_download(repo_id=model_url, allow_patterns=["pytorch_model*"])
        model = load_checkpoint_and_dispatch(
            model, weights_location, device_map=device_map, no_split_module_classes=["T5Block"]
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_url).to(0)
    return model

def get_preds(prompts, tokenizer, model, data_collator,  bs=1, max_new_tokens=20):
    tokenizer.model_max_length = 4096
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
    tokenized_prompts = dataset.map(lambda x: tokenizer(x["prompt"], truncation=True),
                                    remove_columns=dataset.column_names,
                                    batched=True)
    loader = DataLoader(tokenized_prompts, batch_size=bs, shuffle=False, collate_fn=data_collator)
    all_preds = []
    for batch in tqdm(loader):
        outs = model.generate(input_ids=batch["input_ids"].to(0), attention_mask=batch["attention_mask"].to(0),
                temperature=.0, max_new_tokens=max_new_tokens)
        all_preds.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in outs])
    return all_preds