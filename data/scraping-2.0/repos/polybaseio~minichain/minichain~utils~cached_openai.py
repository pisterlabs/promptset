import json
import os 

import numpy as np
import openai
from retry import retry

from minichain.utils.debug import debug
from minichain.utils.disk_cache import disk_cache


def validate_message(message):
    if function := message.get("function_call"):
        try:
            json.loads(function["arguments"])
            return True
        except:
            return False
    return True


@disk_cache
@retry(tries=10, delay=2, backoff=2, jitter=(1, 3))
@debug
def get_openai_response(
    chat_history, functions, model= os.getenv("DEFAULT_OPENAI_MODEL","gpt-3.5-turbo-16k") 
) -> str:  # "gpt-4-0613", "gpt-3.5-turbo-16k"
    messages = []
    for i in chat_history:
        message = i.dict()
        # delete the parent field
        message.pop("parent", None)
        # delete all fields that are None
        message = {k: v for k, v in message.items() if v is not None or k == "content"}
        messages.append(message)
    # print(messages[-2])
    # print("=====================================")
    if len(functions) > 0:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            temperature=0.1,
        )
    else:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
    message = completion.choices[0].message
    response = message.to_dict_recursive()
    # if not validate_message(message):
    #     breakpoint()
    if "error" in messages[-1]["content"].lower():
        print("ERROR detected")
        print(messages[-1])
    return response


@disk_cache
@retry(tries=10, delay=2, backoff=2, jitter=(1, 3))
@debug
def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response["data"][0]["embedding"])
