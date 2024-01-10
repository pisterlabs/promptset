import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time

def make_requests(
        model, 
        messages, 
        max_tokens, 
        temperature, 
        top_p,
        n, 
        stream,  
        frequency_penalty, 
        presence_penalty, 
        stop, 
        logit_bias,
        user,
        retries=3, 
        api_key=None, 
        organization=None
    ):
    response = None
    target_length = max_tokens
    if api_key is not None:
        openai.api_key = api_key
    if organization is not None:
        openai.organization = organization
    retry_cnt = 0
    backoff_time = 10
    messages = [[{"role": "system", "content": "you're the best making {instruction, input, output} data set and the best assistant"}, 
                 {"role": "user", "content": f'{prompt}'}] for prompt in messages]
    
    results = []
    for message in messages:
        retry_cnt = 0
        while retry_cnt <= retries:
            try:
                # openai migration 참고(이전 버전 적용 주의)
                response = openai.chat.completions.create(
                    model=model,
                    messages=message,
                    max_tokens=target_length,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    n=n,
                )
                break
            except openai.error.OpenAIError as e:
                print(f"OpenAIError: {e}.")
                if "Please reduce the length of the messages or completion" in str(e):
                    target_length = int(target_length * 0.8)
                    print(f"Reducing target length to {target_length}, retrying...")
                else:
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                retry_cnt += 1
        results.append(response.choices[0])
    return results