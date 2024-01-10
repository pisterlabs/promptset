import asyncio
import json
import os
import logging
import time
import configparser
import argparse
import tiktoken
import torch
from typing import AsyncIterable, List, Generator, Union, Optional

import requests
import sseclient

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread, BoundedSemaphore
from auto_gptq import exllama_set_max_input_length

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    presence_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    suffix: Optional[str] = None
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    presence_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None

repo_str = 'Nous-Hermes-2-Yi-34B-GPTQ'

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')

# Parse the arguments
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get(repo_str, 'repo')
host = config.get('settings', 'host')

port = args.port if args.port is not None else config.getint('settings', 'port')

# only allow one client at a time
busy = False
condition = asyncio.Condition()

eightbit = False
if repo_str == 'Phind-CodeLlama-34B-v2':
    eightbit = True

torch_dtype = torch.float16  # Set a default dtype
if repo_str == 'zephyr-7b-beta' or repo_str == 'Starling-LM-7B-alpha':
    torch_dtype = torch.float16

revision = "main"
if repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ' or repo_str == 'Yi-34B-Chat-GPTQ':
    revision = 'gptq-4bit-32g-actorder_True'

remote_code = False
if repo_str == 'Nous-Capybara-34B-GPTQ':
    remote_code = True

model = AutoModelForCausalLM.from_pretrained(repo_id,
                                             device_map="auto",
                                             trust_remote_code=remote_code,
                                             revision=revision,
                                             load_in_8bit=eightbit,
                                             torch_dtype=torch_dtype,
                                             use_flash_attention_2=True,
                                             )

max_input_length = 4096
if repo_str == 'Genz-70b-GPTQ' or repo_str == 'Llama-2-70B-chat-GPTQ' or repo_str == 'Yi-34B-Chat-GPTQ' or repo_str == 'Nous-Hermes-2-Yi-34B-GPTQ':
    ## Only for Llama Models
    model = exllama_set_max_input_length(model, 4096)

if repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ' or repo_str == 'Nous-Capybara-34B-GPTQ':
    ## Only for Llama Models
    model = exllama_set_max_input_length(model, 8192)
    max_input_length = 8192

tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False, trust_remote_code=remote_code)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

print("*** Loaded.. now Inference...:")

app = FastAPI(title="Llama70B")

##Use tiktoken for token counts
async def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        message_attributes = vars(message)

        # Iterate over the key-value pairs of the attributes
        for key, value in message_attributes.items():
            num_tokens += len(encoding.encode(str(value)))  # Make sure to convert values to string if they are not already
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with assistant
    return num_tokens


def thread_task(model, inputs, generation_kwargs):
    try:
        model.generate(**generation_kwargs)
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory error caught in thread. Attempting to free up memory.")
        # Move and detach tensors from GPU, then delete inputs
        if torch.is_tensor(inputs):
            if inputs.requires_grad:
                inputs = inputs.detach()
            inputs = inputs.to('cpu')
        del inputs

        torch.cuda.empty_cache()  # Free up unoccupied cached memory

        # Remove the key associated with 'inputs' in generation_kwargs
        if 'inputs' in generation_kwargs:
            del generation_kwargs['inputs']

        time.sleep(5) 
        # Create new inputs and update generation_kwargs
        new_inputs = tokenizer(["USER: say, 'out of memory'\nASSISTANT:"], return_tensors="pt").to("cuda")
        generation_kwargs.update(new_inputs)

        # Retry generation with new input
        model.generate(**generation_kwargs)
        torch.cuda.empty_cache()

    finally:
        # Cleanup after generation is done
        if 'inputs' in locals():  # Check if 'inputs' is still in the local namespace
            del inputs
        torch.cuda.empty_cache()


async def streaming_request(prompt: str, max_tokens: int = 1024, tempmodel: str = 'Llama70', response_format: str = 'completion'):
    """Generator for each chunk received from OpenAI as response
    :param response_format: 'text_completion' or 'chat_completion' to set the output format
    :return: generator object for streaming response from OpenAI
    """
    global busy
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_tokens > max_input_length:
        print(f"Warning: over {max_input_length} tokens in context.")
        busy = False
        yield 'data: [DONE]'
        async with condition:
            condition.notify_all()
        return

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_tokens, temperature=0.01, repetition_penalty=1.1)
    #thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread = Thread(target=thread_task, args=(model, inputs, generation_kwargs))
    thread.start()
    generated_text = ""
    completion_id = f"chatcmpl-{int(time.time() * 1000)}"  # Unique ID for the completion

    if response_format == 'chat_completion':
       yield f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{tempmodel}","choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":"null"}}]}}\n\n'

    for new_text in streamer:
        busy = True
        generated_text += new_text
        reason = None
        if "</s>" in new_text:
            reason = "stop"
            # Strip the </s> from the new_text
            new_text = new_text.replace("</s>", "")

        if "<|end_of_turn|>" in new_text:
            reason = "stop"
            # Strip the </s> from the new_text
            new_text = new_text.replace("<|end_of_turn|>", "")

        if "<|im_end|>" in new_text:
            reason = "stop"
            # Strip the </s> from the new_text
            new_text = new_text.replace("<|im_end|>", "")
        
        if "<|endoftext|>" in new_text:
            reason = "stop"
            # Strip the </s> from the new_text
            new_text = new_text.replace("<|endoftext|>", "")
        
        if response_format == 'chat_completion':
            response_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": tempmodel,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": new_text
                        },
                        "finish_reason": reason
                    }
                ]
            }
        else:  # default to 'completion'
            response_data = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": tempmodel,
                "choices": [
                    {
                        "index": 0,
                        "text": new_text,
                        "logprobs": None,
                        "finish_reason": reason
                    }
                ]
            }
            
        json_output = json.dumps(response_data)
        yield f"data: {json_output}\n\n"  # SSE format

    if response_format == 'chat_completion':
        yield f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{tempmodel}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n'
    else:
        yield 'data: [DONE]'

    busy = False
    async with condition:
        condition.notify_all()

def non_streaming_request(prompt: str, max_tokens: int = 1024, tempmodel: str = 'Llama70', response_format: str = 'completion'):

    # Assume generated_text is the output text you want to return
    # and assume you have a way to calculate prompt_tokens and completion_tokens
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    #print("Prompt Tokens: " + str(prompt_tokens))

    generated_text = ''
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            temperature=0.01,
            #top_p=0.95,
            #top_k=40,
            repetition_penalty=1.1,
        )
        output = pipe(prompt, return_full_text=False)
        generated_text = output[0]['generated_text']
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory error caught. Attempting to free up memory.")
        torch.cuda.empty_cache()  # Free up unoccupied cached memory
        generated_text = "out of memory"

    completion_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
    full_tokens = completion_tokens + prompt_tokens

    # Prepare the response based on the format required
    if response_format == 'completion':
        response_data = {
            "id": "cmpl-0",
            "object": "text_completion",
            "created": int(time.time()),
            "model": tempmodel,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": full_tokens
            }
        }
    elif response_format == 'chat_completion':
        response_data = {
            "id": "chatcmpl-0",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": tempmodel,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": full_tokens
            }
        }
    else:
        raise ValueError(f"Unsupported response_format: {response_format}")

    return response_data

@app.post('/v1/completions')
async def main(request: CompletionRequest):

    global busy
    async with condition:
        while busy:
            await condition.wait()
        busy = True

    try:
        prompt = ""
        if isinstance(request.prompt, list):
            # handle list of strings
            prompt = request.prompt[0]  # just grabbing the 0th index
        else:
            # handle single string
            prompt = request.prompt

        if request.stream:
            response = StreamingResponse(streaming_request(prompt, request.max_tokens, tempmodel=repo_str), media_type="text/event-stream")
        else:
            response_data = non_streaming_request(prompt, request.max_tokens, tempmodel=repo_str)
            response = response_data  # This will return a JSON response
    
    except Exception as e:
        # Handle exception...
        async with condition:
            if request.stream == True:
                busy = False
                await condition.notify_all()

    finally:
        async with condition:
            if request.stream != True:
                busy = False
                condition.notify_all()

    return response

async def format_prompt(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"{message.content}\n\n"
        elif message.role == "user":
            formatted_prompt += f"### User:\n{message.content}\n\n"
        elif message.role == "assistant":
            formatted_prompt += f"### Assistant:\n{message.content}\n\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "### Assistant:\n"
    return formatted_prompt

async def format_prompt_yi(messages):
    formatted_prompt = ""
    system_message_found = False
    
    # Check for a system message first
    for message in messages:
        if message.role == "system":
            system_message_found = True
            break
    
    # If no system message was found, prepend a default one
    if not system_message_found:
        formatted_prompt = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            formatted_prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|im_start|>assistant\n"
    return formatted_prompt

async def format_prompt_nous(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"{message.content}\n\n"
        elif message.role == "user":
            formatted_prompt += f"USER: {message.content}\n"
        elif message.role == "assistant":
            formatted_prompt += f"ASSISTANT: {message.content}\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "ASSISTANT: "
    return formatted_prompt

async def format_prompt_code(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"### System Prompt\nYou are an intelligent programming assistant.\n\n"
        elif message.role == "user":
            formatted_prompt += f"### User Message\n{message.content}\n\n"
        elif message.role == "assistant":
            formatted_prompt += f"### Assistant\n{message.content}\n\n"
    # Add the final "### Assistant" with ellipsis to prompt for the next response
    formatted_prompt += "### Assistant\n..."
    return formatted_prompt

async def format_prompt_zephyr(messages):
    formatted_prompt = ""
    for message in messages:
        if message.role == "system":
            formatted_prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            formatted_prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            formatted_prompt += f"<|assistant|>\n{message.content}</s>\n"
    # Add the final "### Assistant:\n" to prompt for the next response
    formatted_prompt += "<|assistant|>\n"
    return formatted_prompt

async def format_prompt_starling(messages):
    formatted_prompt = ""
    system_message = ""
    for message in messages:
        if message.role == "system":
            # Save system message to prepend to the first user message
            system_message += f"{message.content}\n\n"
        elif message.role == "user":
            # Prepend system message if it exists
            if system_message:
                formatted_prompt += f"GPT4 Correct User: {system_message}{message.content}<|end_of_turn|>"
                system_message = ""  # Clear system message after prepending
            else:
                formatted_prompt += f"GPT4 Correct User: {message.content}<|end_of_turn|>"
        elif message.role == "assistant":
            formatted_prompt += f"GPT4 Correct Assistant: {message.content}<|end_of_turn|>"  # Prep for user follow-up
    formatted_prompt += "GPT4 Correct Assistant: \n\n"
    return formatted_prompt

async def format_prompt_mixtral(messages):
    formatted_prompt = "<s> "
    system_message = ""
    for message in messages:
        if message.role == "system":
            # Save system message to prepend to the first user message
            system_message += f"{message.content}\n\n"
        elif message.role == "user":
            # Prepend system message if it exists
            if system_message:
                formatted_prompt += f"[INST] {system_message}{message.content} [/INST] "
                system_message = ""  # Clear system message after prepending
            else:
                formatted_prompt += f"[INST] {message.content} [/INST] "
        elif message.role == "assistant":
            formatted_prompt += f" {message.content}</s> "  # Prep for user follow-up
    return formatted_prompt

@app.post('/v1/chat/completions')
async def mainchat(request: ChatCompletionRequest):

    global busy
    async with condition:
        while busy:
            await condition.wait()
        busy = True

    try:
        t = await num_tokens_from_messages(request.messages)
        print(t)
 
        prompt = ''
        if repo_str == 'Phind-CodeLlama-34B-v2':
            prompt = await format_prompt_code(request.messages)
        elif repo_str == 'zephyr-7b-beta':
            prompt = await format_prompt_zephyr(request.messages)
        elif repo_str == 'Starling-LM-7B-alpha':
            prompt = await format_prompt_starling(request.messages)
        elif repo_str == 'Mixtral-8x7B-Instruct-v0.1-GPTQ':
            prompt = await format_prompt_mixtral(request.messages)
        elif repo_str == 'Yi-34B-Chat-GPTQ' or repo_str == 'Nous-Hermes-2-Yi-34B-GPTQ':
            prompt = await format_prompt_yi(request.messages)
        elif repo_str == 'Nous-Capybara-34B-GPTQ':
            prompt = await format_prompt_nous(request.messages)
        else:
            prompt = await format_prompt(request.messages)
        print(prompt)

        if request.stream:
            response = StreamingResponse(streaming_request(prompt, request.max_tokens, tempmodel=repo_str, response_format='chat_completion'), media_type="text/event-stream")
        else:
            response_data = non_streaming_request(prompt, request.max_tokens, tempmodel=repo_str, response_format='chat_completion')
            response = response_data  # This will return a JSON response
    
    except Exception as e:
        # Handle exception...
        async with condition:
            if request.stream == True:
                busy = False
                await condition.notify_all()

    finally:
        async with condition:
            if request.stream != True:
                busy = False
                condition.notify_all()

    return response




@app.get('/ping')
async def get_status():
    return {"ping": "pong"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="debug")
