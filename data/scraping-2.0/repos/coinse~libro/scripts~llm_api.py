import os
import json
import requests
import random

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
HF_KEY = os.getenv("HF_API_KEY")

AVAILABLE_MODEL_INFO = {
    'OpenAI/text-curie-001': {
        'query_type': 'openai',
        'uses_chat': False,
    },
    'OpenAI/text-davinci-002': {
        'query_type': 'openai',
        'uses_chat': False,
    },
    'OpenAI/text-davinci-003': {
        'query_type': 'openai',
        'uses_chat': False,
    },
    'OpenAI/gpt-3.5-turbo': {
        'query_type': 'openai',
        'uses_chat': True,
    },
    'OpenAI/gpt-4': {
        'query_type': 'openai',
        'uses_chat': True,
    },
    'bigscience/bloom': {
        'query_type': 'hf_hosted',
        'uses_chat': False,
    },
    'bigscience/bloom-7b1': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'bigscience/bloom-3b': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'bigscience/bloomz': {
        'query_type': 'hf_hosted',
        'uses_chat': False,
    },
    'bigscience/bloomz-7b1': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'bigscience/bloomz-3b': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'facebook/incoder-6B': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'facebook/incoder-1B': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'Salesforce/codegen-16B-multi': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'Salesforce/codegen-6B-multi': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'Salesforce/codegen-2B-multi': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'Salesforce/codegen-350M-multi': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'EleutherAI/gpt-neox-20b': {
        'query_type': 'hf_hosted',
        'uses_chat': False,
    },
    'EleutherAI/gpt-neo-2.7b': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'EleutherAI/gpt-neo-1.3b': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'EleutherAI/gpt-neo-125m': {
        'query_type': 'huggingface',
        'uses_chat': False,
    },
    'Databricks/dolly-v2-12b': {
        'query_type': 'unknown',
        'uses_chat': False,
    },
    'BlinkDL/rwkv-4-pile-14b': {
        'query_type': 'unknown',
        'uses_chat': False,
    },
    'BlinkDL/rwkv-4-pile-7b': {
        'query_type': 'unknown',
        'uses_chat': False,
    },
    'BlinkDL/rwkv-4-raven': {
        'query_type': 'unknown',
        'uses_chat': True,
    },
    'Salesforce/codegen2-16B': {
        'query_type': 'self_hosted',
        'uses_chat': False,
    }
}
AVAILABLE_MODELS = AVAILABLE_MODEL_INFO.keys() # just for clean code

TEMP = 0.7

# Helper functions

def model_is_chat(model):
    return AVAILABLE_MODEL_INFO[model]['uses_chat']

def tiny_noise(scale=1/1000):
    return scale*random.random()-0.5*scale

# Query functions
def query_hf_hosted_llm(prompt, model, stop_tokens, use_cache=False, end_len=1000*4):
    def single_query(wip_prompt):
        use_temp = TEMP if use_cache else (TEMP + tiny_noise())
        data = json.dumps({'inputs': wip_prompt, 'parameters': {'temperature': use_temp}})
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {HF_KEY}", "Content-Type": "application/json"}
        response = requests.request("POST", API_URL, headers=headers, data=data)
        assert response.status_code == 200, f'Response status was non-normal ({response.status_code}): {response.content}'
        return json.loads(response.content.decode("utf-8"))[0]['generated_text']
    
    new_content = ''
    while (all(t not in new_content for t in stop_tokens) and 
           len(new_content) < end_len):
        full_gen_text = single_query(prompt + new_content)
        if full_gen_text == prompt+new_content: # predicting end token
            return new_content
        new_content = full_gen_text[len(prompt):]
    earliest_stop_loc = min([len(new_content)] + 
                            [new_content.index(t) for t in stop_tokens if new_content.index(t) >= 0])
    return new_content[:earliest_stop_loc]

def query_self_hosted_llm(prompt, stop_tokens, max_tokens=256):
    payload = {
        'text': prompt,
        'max_new_tokens': max_tokens,
    }
    json_payload = json.dumps(payload)
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:23623', data=json_payload, headers=headers)
    gen_content = response.json()['result']
    earliest_stop_loc = min([len(gen_content)] +
                            [gen_content.index(t) for t in stop_tokens if t in gen_content])
    return gen_content[:earliest_stop_loc]

def query_chat_llm(prompt, model, stop_tokens):
    assert model in AVAILABLE_MODEL_INFO, f'Unknown model {model}'
    model_info = AVAILABLE_MODEL_INFO[model]
    assert type(prompt) == list
    assert model_info['uses_chat']

    if model_info['query_type'] == 'openai':
        base_model_name = model.split('/')[-1]
        response = openai.ChatCompletion.create(
            model=base_model_name,
            messages=prompt, # chat-style prompt
            n=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_tokens
        )
        gen_result = response["choices"][0]["message"]["content"]
        if "```" in gen_result:
            gen_result = gen_result.split("```")[1]
            gen_result = gen_result.removeprefix('java')
    elif model_info['query_type'] == 'hf_hosted':
        raise NotImplementedError
    else:
        raise NotImplementedError(f'Unknown query type {model_info["query_type"]}')
    return gen_result

def query_string_llm(prompt, model, stop_tokens):
    assert model in AVAILABLE_MODEL_INFO, f'Unknown model {model}'
    model_info = AVAILABLE_MODEL_INFO[model]
    assert type(prompt) == str
    assert not model_info['uses_chat']

    if model_info['query_type'] == 'openai':
        base_model_name = model.split('/')[-1]
        response = openai.Completion.create(
            engine=base_model_name,
            prompt=prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_tokens
        )
        gen_result = response['choices'][0]['text']
    elif model_info['query_type'] == 'hf_hosted':
        gen_result = query_hf_hosted_llm(prompt, model, stop_tokens)
    elif model_info['query_type'] == 'self_hosted':
        gen_result = query_self_hosted_llm(prompt, stop_tokens, max_tokens=256)
    else:
        raise NotImplementedError(f'Unknown query type {model_info["query_type"]}')
    return gen_result

def query_llm(prompt, model, stop_tokens):
    # sanity checks
    assert model in AVAILABLE_MODELS, f'Unknown model {model}'
    model_info = AVAILABLE_MODEL_INFO[model]
    if model_info['uses_chat']:
        assert type(prompt) == list
    else:
        assert type(prompt) == str
    
    # actual execution
    query_func = query_chat_llm if model_info['uses_chat'] else query_string_llm
    return query_func(prompt, model, stop_tokens)
