import openai
import os
import json
model_list = {
    "completion": {
        "text-davinci-003": 4096,
        "davinci": 2049,
        "code-davinci-002": 8000,
        "text-curie-001": 2049, 
        "gpt-3": 4096
    },
    "chat": {
        "gpt-3.5-turbo": 4095,
        "gpt-4": 8191,
        "gpt-4-32k": 32767
    },
    "other": {
        "other": 2048
    }
}

def _gpt_new_max_token(model:str, max_token:int=float("inf")):
    """ Get max token allowed for model
    """
    # flatten out model list
    max_token_list = {}
    for _, models_by_endpoint in model_list.items():
        max_token_list.update(models_by_endpoint)
    # if model not found, use "other" as model aname
    if model not in max_token_list.keys():
        model = "other"
        
    # overlimit, update with max_token
    if max_token > max_token_list[model]:
        return max_token_list[model]
    # safe range, return as is
    else:
        return max_token
        
def _openai_gpt_completion_request(prompt:list|str, context:str="", max_token:int=516, engine:str="text-davinci-003", temperature:float=0.5, top_p:int=1, frequency_penalty:float=0, presence_penalty:float=0, organization_token:str = "", connection_token:str = "", azure_api_base:str="", azure_api_version:str="2022-12-01"):
    """
    Make GPT completion request
    """
    # make sure context have at least two linebreak at the end
    if context[-1] != "\n":
        context += "\n"
        if context[-2] != "\n":
            context += "\n"
    # request
    if isinstance(prompt, str):
        prompt=context + prompt,
    else:
        prompt=context + prompt[-1]["content"]
    GPT_result = openai.Completion.create(
            engine=engine, 
            prompt=prompt,
            max_tokens=_gpt_new_max_token(engine, max_token),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
    
    return GPT_result

def _openai_gpt_chat_request(prompt:list, context:str="", max_token:int=516, engine:str="gpt-4", temperature:float=0.5, top_p:int=1, frequency_penalty:float=0, presence_penalty:float=0):
    """
    Make GPT chat request
    """
    # request
    if isinstance(prompt, str):
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
    else:
        messages=[
            {"role": "system", "content": context},
            *prompt
        ]
    GPT_result = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        max_tokens=_gpt_new_max_token(engine, max_token),
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return GPT_result
    
def gpt_list_to_chat(chat_list:list):
    messages = []
    for i, message in enumerate(chat_list):
        if i % 2 == 0:
            messages.append({"role": "user", "content": message})
        elif i % 2 == 1:
            messages.append({"role": "assistant", "content": message})
    return messages

def gpt_request(prompt:list, context:str="", max_token:int=516, engine:str="gpt-4", temperature:float=0.5, top_p:int=1, frequency_penalty:float=0, presence_penalty:float=0, connection_token:str=""):
    openai.api_key =  connection_token if connection_token else os.getenv("GPT_API_KEY")
    if not openai.api_key:
        raise AttributeError("No GPT_API_KEY set")
    if engine in model_list["chat"]:
        return (_openai_gpt_chat_request(prompt, context, max_token, engine, temperature, top_p, frequency_penalty, presence_penalty), "chat")
    elif engine in model_list["completion"]:
        return (_openai_gpt_completion_request(prompt, context, max_token, engine, temperature, top_p, frequency_penalty, presence_penalty), "completion")
    else:
        raise AttributeError("Unknown engine")