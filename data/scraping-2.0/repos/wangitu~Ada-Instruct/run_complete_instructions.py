import sys
import os
import json
import fire
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from typing import Sequence, Optional

from async_api import OpenAIMultiClient, legacy_api
from key_manager import OpenaiAPIKeyPool, Empty
from tasks import get_task


@dataclass
class OpenAIDecodingArguments:
    """These are the values used in the WizardCoder paper"""
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 2048
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    
    
def complete_synthesized_instructions(task, synthesized_instructions, key_pool, api):
    for synthesized_instruction in synthesized_instructions:
        try:
            api.request(
                data=task.fill_request_for_completion(synthesized_instruction),
                metadata={
                    "original_id": synthesized_instruction["id"],
                    "original_instruction": synthesized_instruction["instruction"],
                    **{k: v for k, v in synthesized_instruction.items() if k not in ["id", "instruction"]}
                },
                api_key=key_pool.random()
            )
            
        except Empty:
            print("No available openai api key")
            return
        
def dump_completions(completions, out_file):
    total = len(completions)
    valid = len([completion for completion in completions if completion["valid"]])
    completions_to_dump = {
        "total": total,
        "valid": valid,
        "completions": completions
    }
    
    out_dir, _ = os.path.split(out_file)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(completions_to_dump, f, ensure_ascii=False, indent=2)
    
    
def complete_synthesized_instruction_set(
    task_name: str,
    in_file: str,
    out_file: str,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    stop: Optional[Sequence[str]] = None,
    openai_key_file: str = "./openai_keys.json",
    wait_interval: int = 20,
    dump_frequency: int = 50
):
    task = get_task(task_name)  
    
    with open(in_file, 'r', encoding='utf-8') as f:
        systhesized = json.load(f)
    
    completions = []
    # resume from generation
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            completions = json.load(f)["completions"]
            # completions = [completion for completion in completions if completion["valid"]]
        completed_ids = set(completion["original_id"] for completion in completions)
        systhesized = [i for i in systhesized if i["id"] not in completed_ids]
    
    decoding_args = OpenAIDecodingArguments(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop
    )
    
    key_pool = OpenaiAPIKeyPool(openai_key_file)
    
    api = OpenAIMultiClient(
        concurrency=len(key_pool.available),
        wait_interval=wait_interval,
        data_template={
            "model": model_name,
            **vars(decoding_args)
        },
        custom_api=partial(legacy_api) # you may invoke proxy by ```custom_api=partial(legacy_api, proxy="http://your_proxy")```
    )
    
    api.run_request_function(complete_synthesized_instructions, task, systhesized, key_pool, api)
    
    for response in tqdm(api, total=len(systhesized)):
        if response.failed:
            if "your account" in response.response.lower() or "you exceed" in response.response.lower():
                key_pool.delete(response.api_key)
            continue
        
        content = response.response["choices"][0]["message"]["content"]
        result = task.postprocess_completion(
            synthesized_instruction={
                "id": response.metadata["original_id"],
                "instruction": response.metadata["original_instruction"]
            }, 
            content=content
        )
        completions.append({
            "valid": result is not None,
            "instruction": result["instruction"] if result is not None else response.metadata["original_instruction"],
            "output": result["output"] if result is not None else content,
            **response.metadata,
            **({k: v for k, v in result.items() if k not in ["instruction", "output"]} if result is not None else {})
        })
        
        # return `api_key` to `key_pool` for reuse
        key_pool.put(response.api_key)
        
        if len(completions) % dump_frequency == 0:
            dump_completions(completions, out_file)
            key_pool.dump()
    
    dump_completions(completions, out_file)
    key_pool.dump()
        

if __name__ == '__main__':
    fire.Fire(complete_synthesized_instruction_set)
