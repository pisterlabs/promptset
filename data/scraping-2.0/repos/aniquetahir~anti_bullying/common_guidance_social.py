import pickle
import threading

import guidance
from time import sleep
import timeit
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
# from common_generative import generate_prompt, model, tokenizer
from typing import List, Dict, Union
from collections import deque
from instagram.label_instagram import get_all_prompts
import numpy as np
import math
import fire
from tqdm import tqdm
guidance.llms._llm

gpu_llms = {}
q_open_content = deque()
q_structured_content = deque()

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:\n"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:\n"""
response_format = """
a) The post with the active ID ({{gen 'postid' max_tokens=10 stop=')' pattern='[0-9]+'}}) {{gen 'summary' max_tokens=100 temperature=0 stop='b)'}}
b) {{select 'bullying' options=['Yes', 'No']}}, the active post is {{gen 'bullyingextra' temperature=0 max_tokens=100 stop='c)'}}
c) {{select 'antibullying' options=['Yes', 'No']}}, the active post is {{gen 'antibullyingextra' temperature=0 max_tokens=100 stop='d)'}}
d) The reason the post is {{select 'reasonbull1' options=['not considered', 'considered']}} bullying is because {{gen 'reasonbull2' temperature=0 max_tokens=200 stop='e)'}}
e) The reason the post is {{select 'reasonantibull1' options=['not considered', 'considered']}} anti-bullying is because {{gen 'reasonantibull2' temperature=0 max_tokens=200 stop='f)'}}
f) Whether the post is neither bullying or anti-bullying can be determined by considering answers to (b) and (c). Thus, {{gen 'reasonnei2' temperature=0 max_tokens=200 stop='.'}}
"""


# model = AutoModelForCausalLM.from_pretrained("/media/anique/Data/projects/llama-weights/llama2-7B-merged",
#                                              device_map='auto', load_in_8bit=True)
# # use better transformer
# model.to_bettertransformer()
# model.tie_weights()
# # compile model
# model = torch.compile(model)
#
# tokenizer = AutoTokenizer.from_pretrained("/media/anique/Data/projects/llama-weights/llama2-7B", padding=True, padding_side='left')
# tokenizer.pad_token = tokenizer.eos_token

# guidance.llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer)
model_path = "/media/anique/Data/projects/llama-weights/llama2-7B"
tokenizer_path = "/media/anique/Data/projects/llama-weights/llama2-7B"
# model_path = "/scratch/artahir/llama/hf_weights/llama2-13B-phase2-aug"
# tokenizer_path = "/scratch/artahir/llama/hf_weights/llama2-13B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

def extract_answers(resp):
    return {
        'postid': resp['postid'],
        'summary': resp['summary'],
        'bullying': resp['bullying'].strip(),
        'bullyingextra': resp['bullyingextra'].strip(),
        'antibullying': resp['antibullying'].strip(),
        'antibullyingextra': resp['antibullyingextra'].strip(),
        'reasonbull1': resp['reasonbull1'],
        'reasonbull2': resp['reasonbull2'],
        'reasonantibull1': resp['reasonantibull1'],
        'reasonantibull2': resp['reasonantibull2'],
        'reasonnei2': resp['reasonnei2']
    }


def convert_response(guidance_prompt: str, llm) -> Dict[str, Union[str, bool]]:
    json_program = guidance(guidance_prompt, llm=llm, silent=True, stream=False, caching=False)
    guidance_response = json_program()
    return guidance_response

def gpu_loop(gpu_id):
    llm = gpu_llms[gpu_id]
    while True:
        try:
            content_to_process = q_open_content.popleft()
        except IndexError:
            sleep(1)
            continue
        try:
            structure_response = convert_response(content_to_process, llm)
            q_structured_content.append((content_to_process, extract_answers(structure_response)))
        except Exception as e:
            q_structured_content.append((content_to_process, {'error': str(e)}))


async def _get_guided_responses_batched(prompts):
    batch_executions = []
    for prompt in prompts:
        prompt = prompt + response_format
        resp_program = guidance(prompt,
                silent=True,
                async_mode=True,
                stream=False,
            )
        response_fn = resp_program(
            silent=True,
            async_mode=True,
            stream=False,
        )
        batch_executions.append(response_fn)
    responses = await asyncio.gather(*batch_executions)
    return responses

def get_guided_responses_batched(prompts, batch_size=10):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        results.extend(asyncio.run(_get_guided_responses_batched(prompts[i:i+batch_size])))
    return results

def process_que(prompts, part=-1):
    for prompt in prompts:
        q_open_content.append(prompt + response_format)
    while len(q_structured_content) < len(prompts):
        sleep(1)

    if len(q_structured_content)>100 and len(q_structured_content) % 100 == 0:
        print(f"Processed {len(q_structured_content)} prompts")
        # save the results in a cache
        with open(f'cache_part_{part}.pkl', 'w') as response_file:
            pickle.dump(list(q_structured_content), response_file)

    return [q_structured_content.popleft() for _ in range(len(prompts))]

def get_instagram_responses(part: int, total_parts: int):
    instagram_prompts = get_all_prompts()
    # sort the prompts
    instagram_prompts = sorted(instagram_prompts)
    samples_per_part = math.ceil(len(instagram_prompts) / total_parts)

    # get the start and end index for this part
    start_index = (part - 1) * samples_per_part
    end_index = min(part * samples_per_part, len(instagram_prompts))

    # filter the prompts
    prompts = instagram_prompts[start_index:end_index]
    results = get_guided_responses_batched(prompts)
    # process the prompts
    # results = process_que(prompts, part=part)

    # save the results
    with open(f'instagram_responses_part_{part}.pkl', 'w') as respones_file:
        pickle.dump(results, respones_file)

    return process_que(prompts)


if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
                                                 # load_in_8bit=True)
    # use better transformer
    model.to_bettertransformer()
    model.tie_weights()
    # compile model
    model = torch.compile(model)
    class TTransformers(guidance.llms.Transformers):
        llm_name = 'transformer_0'
    guidance.llm = TTransformers(model=model,
                                             tokenizer=tokenizer, device_map="auto")
    # create an llm for each gpu
    # for i in range(torch.cuda.device_count()):
    #
    #     model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                                  load_in_8bit=True, device_map='auto')
    #     # use better transformer
    #     model.to_bettertransformer()
    #     model.tie_weights()
    #     # compile model
    #     model = torch.compile(model)
    #     gpu_llms[i] = guidance.llms.Transformers(model=model,
    #                                              tokenizer=tokenizer, device_map={"": f'cuda:{i}'})
    #     gpu_llms[i].llm_name = f'cuda:{i}'
    #
    # for i in range(torch.cuda.device_count()):
    #     threading.Thread(target=gpu_loop, args=(i,)).start()

    get_instagram_responses(fire.Fire(get_instagram_responses))

    # time the execution
    # with open('../../projects/alpaca-lora/bard_data_alpaca.json', 'r') as bard_data_file:
    #     bard_data = json.load(bard_data_file)
    #
    # samples = [generate_prompt(x['instruction'], x['input']) for x in bard_data[:20]]
    # process_que(samples)
    # print(len(samples))
    exit()