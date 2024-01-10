import requests
import traceback
import os
import re
import gc 
import openai
import fire
import torch
import traceback
import subprocess
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from .validatecode import validate_code
from .prompts import USER_PROMPT, USER_PROMPT_LLAMA
from llama import Llama

def get_api_key():
    api_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = api_key    
    return api_key

api_key = get_api_key()

def print_gpu_memory():
    """Utility function to print GPU memory usage."""
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MiB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MiB")

def free_gpu_memory():
    torch.cuda.empty_cache()  
    gc.collect()  

def monitor_gpu_stats():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
        "--format=csv",
        "-l", 
        "1"
    ]
    with open("gpu_stats.csv", "w") as f:
        subprocess.Popen(cmd, stdout=f)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def generate_gpt4_response(
        prompt, 
        modality, 
        api_key, 
        ckpt_dir, 
        tokenizer_path, 
        max_seq_len=256, # max length of the input sequence
        max_gen_len=None, # Optional
        max_batch_size=4, 
        temperature=0, # 0 implies models will always pick the most likely next word, ignoring top_p re llama-2, aligns with gpt-4
        top_p=0.9 # overridden here by temperature=0, aligns with gpt-4
    ):
    try:
        print("Prompt: ", prompt)   # print prompt
        print("Modality: ", modality)   # print api key

        monitor_gpu_stats()

        is_valid = validate_code(prompt, modality, api_key, max_seq_len)
        
        if not is_valid:
            raise ValueError(f"Invalid code: {prompt}")
        
        user_prompt = USER_PROMPT.format(modality=modality, prompt=prompt)
        user_prompt_llama = USER_PROMPT_LLAMA.format(prompt=prompt)
        
        openai.api_key = api_key

        if modality == "gpt-4":
            def create_chat_completion():
                return openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_seq_len, 
                    n=1,
                    stop=None,
                    temperature=0,
                )

            response = create_chat_completion()

            print(modality, " response:", response)  # print statements to see the values of variables and the response from the GPT-4 API

            final_response = response['choices'][0]['message']['content']
            return final_response.strip()
        
        elif modality == "codellama-7b":
            print("Before setup:")
            print(torch.cuda.is_available())
            print_gpu_memory()  
            free_gpu_memory()  

            setup(0, 1)  # initialize the process group with rank=0, world_size=1

            print("After setup, before model build:")
            print_gpu_memory()  

            generator = Llama.build(
                ckpt_dir=ckpt_dir,
                tokenizer_path=tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )

            print("After model build, before text generation:")
            print_gpu_memory()  

            prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """\
import socket

def ping_exponential_backoff(host: str):""",
        """\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":"""
            ]

            try:
                results = generator.text_completion(
                    prompts,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                for prompt, result in zip(prompts, results):
                    print(prompt)
                    print(f"> {result['generation']}")
                    print("\n==================================\n")

            except Exception as e:
                print(f"An error of type {type(e).__name__} occurred during the generation: {str(e)}")
                traceback.print_exc()
                return str(e)

            finally:
                print("Before cleanup:")
                print_gpu_memory() 

                cleanup()  # cleanup the process group

                print("After cleanup:")
                print_gpu_memory()  

    # part of try/except block, here the except block catches error and prints it to the terminal
    except Exception as e:
        print(f"An error of type {type(e).__name__} occurred during the generation: {str(e)}")
        traceback.print_exc()
        return str(e)
