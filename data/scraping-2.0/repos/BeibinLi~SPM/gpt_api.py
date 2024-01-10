import copy
import hashlib
import os
import re
import time
import requests
import json

import diskcache
import numpy as np
import openai
from nltk.stem import PorterStemmer
from termcolor import colored

api_type = "azure"
api_base = "https://msrcore.openai.azure.com/"
#api_base = "https://gcrgpt4aoai4.openai.azure.com/"
#api_base = os.getenv("PROXY_OPENAI_ENDPOINT")
api_version = "2023-03-15-preview"
api_key = os.getenv("CORE_AZURE_KEY").strip().rstrip()
#api_key = os.getenv("CORE_AZURE_KEY_GPT_4").strip().rstrip()
#api_key = os.getenv("PROXY_OPENAI_KEY").strip().rstrip()

ps = PorterStemmer()


def persistent_hash(obj):
    # Convert the input object to a string representation
    obj_str = repr(obj).encode('utf-8')
    # Calculate the SHA-256 hash of the string representation
    hash_obj = hashlib.sha256(obj_str).hexdigest()
    # Return the hash as a string
    return hash_obj


def stem_and_hash(sentence):
    # remove all non-alphanumeric characters
    sentence = re.sub(r'[^a-zA-Z0-9]', ' ', copy.deepcopy(sentence))

    # all whitespaces to one space
    sentence = re.sub(r"\s+", " ", sentence)

    sentence = sentence.lower().split()

    stemmed = [ps.stem(s) for s in sentence if len(s)]

    stemmed = "".join(stemmed)

    return persistent_hash(stemmed)


def cache_llm_infer_result(func):
    cache_data = diskcache.Cache(".diskcache")    # setup the cache database

    def wrapper(*args, **kwargs):

        # remove the "self" or "cls" in the args
        key = f"{func.__name__}_{stem_and_hash(str(args[1:]) + str(kwargs))}"

        # Do not use cache for local models.
        use_cache = True
        if "model" in kwargs and kwargs["model"] in ["tuned", "origin"]:
            use_cache = False
        if "origin" in args or "tuned" in args:
            use_cache = False

        storage = cache_data.get(key, None)
        if storage and use_cache:
            return storage["result"]

        # Otherwise, call the function and save its result to the cache file
        result = func(*args, **kwargs)

        storage = {"result": result, "args": args[1:], "kwargs": kwargs}

        if use_cache:
            cache_data.set(key, storage)

        return result

    return wrapper


def handle_prev_message_history(agent_name, msg, prev_msgs):
    """
    Prepare and handle the message history, ensuring the presence of a system message.

    This function takes the current agent's name, message, and the previous messages
    history. It ensures that if there's no system message in the previous messages,
    one is added. Then, it appends the current message to the history.

    Parameters:
    - agent_name (str): The name of the current agent sending the message.
    - msg (str): The content of the current message.
    - prev_msgs (list of tuple): A list of previous messages, where each message is
                                 represented as a tuple (agent_name, content).

    Returns:
    - list: A list of messages (in dictionary format) with the format:
            {
                "role": agent_name,
                "content": message_content
            }

    Note:
    The function ensures that a system message with content "You are an AI assistant
    that writes Python code to answer questions." is present at the beginning of the
    history if not already present.

    Example:
    Given agent_name="User", msg="Hello", and prev_msgs=[("system", "system_msg"),
    ("User", "Hi")], the function will return:
    [
        {"role": "system", "content": "system_msg"},
        {"role": "User", "content": "Hi"},
        {"role": "User", "content": "Hello"}
    ]
    """
    if prev_msgs is None:
        prev_msgs = []

    if "system" in [_agent_name for _agent_name, _message in prev_msgs]:
        messages = []
    else:
        # Handle missing system message for ChatCompletion
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that writes Python code to "
                           "answer questions."
            },
        ]

    if prev_msgs and len(prev_msgs):
        messages += [{
            "role": _agent_name,
            "content": _message
        } for _agent_name, _message in prev_msgs]

    return messages + [{"role": agent_name, "content": msg}]


class AzureGPTClient():

    def __init__(self):
        #print("OpenAI Endpoint:", openai.api_base)
        #print("OpenAI Key:", openai.api_key)
        pass

    @cache_llm_infer_result
    def reply(self,
              agent_name,
              msg,
              num_response=1,
              stop=None,
              model="gpt-4",
              prev_msgs=None,
              temperature=0,
              top_p=1,
              max_tokens=1000):
        while True:    # Run until succeed
            time_to_sleep = 5    # seconds between tries
            try:
                return self._try_reply(agent_name=agent_name,
                                       msg=msg,
                                       num_response=num_response,
                                       stop=stop,
                                       model=model,
                                       prev_msgs=prev_msgs,
                                       temperature=temperature,
                                       top_p=top_p,
                                       max_tokens=max_tokens)
            except openai.error.RateLimitError as e:
                print(e)
                time_to_sleep = int(
                    re.findall(r"retry after (\d+) second", str(e))[0])
                print(colored("(Azure) Rate limit exceeded.", "yellow"))
            except openai.error.APIConnectionError as e:
                print(e)
                print(colored("(Azure) API connection error.", "yellow"))
            except openai.error.Timeout as e:
                print(e)
                print(colored("(Azure) API Timed Out.", "yellow"))
            except openai.error.APIError as e:
                print(e)
                print(colored("(Azure) API Error.", "yellow"))

            print(colored(f"Sleeping {time_to_sleep} seconds...", "yellow"))
            time.sleep(time_to_sleep)

    def _try_reply(self,
                   agent_name,
                   msg,
                   num_response=1,
                   stop=None,
                   model="gpt-4",
                   prev_msgs=None,
                   temperature=0,
                   top_p=1,
                   max_tokens=1000):
        if num_response > 1:
            assert temperature > 0 or top_p < 1

        messages = handle_prev_message_history(agent_name, msg, prev_msgs)

        if model in ["tuned", "origin"]:
            # we are going to use our own model
            url = 'http://localhost:5000/chat'

            data = {
                "message": msg,
                "model": model,
                "n": num_response,
                "temperature": temperature,
                "top_p": top_p,
                "messages": prev_msgs,
                "max_tokens": max_tokens,
                "secret": "API KEY"
            }
            headers = {"Content-Type": "application/json"}

            response = requests.post(url,
                                     headers=headers,
                                     data=json.dumps(data))

            # print(colored(response, "red"))
            ans = response.json()["answers"]
            # print("Bot:", colored(ans, "green"))
            return ans

        if "gpt-4" in model or "turbo" in model:
            response = openai.ChatCompletion.create(engine=model,
                                                    messages=messages,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    stop=None,
                                                    n=num_response,
                                                    max_tokens=max_tokens)
            answers = [
                response["choices"][i]["message"]["content"]
                for i in range(len(response["choices"]))
            ]
        else:
            response = openai.Completion.create(engine=model,
                                                prompt=msg,
                                                temperature=temperature,
                                                top_p=top_p,
                                                stop=stop,
                                                n=num_response,
                                                max_tokens=max_tokens)

            answers = [
                response["choices"][i]["text"]
                for i in range(len(response["choices"]))
            ]

        # print(colored("response:", "green"), response)
        #print(colored("ans:", "green"), answers)

        # pdb.set_trace()
        return answers


def cache_func_call(func):
    cache_data = diskcache.Cache(".diskcache")    # setup the cache database

    def wrapper(*args, **kwargs):
        key = f"{func.__name__}_{stem_and_hash(str(args) + str(kwargs))}"

        storage = cache_data.get(key, None)
        if storage:
            return storage["result"]

        # Otherwise, call the function and save its result to the cache file
        result = func(*args, **kwargs)

        storage = {"result": result, "args": args, "kwargs": kwargs}
        cache_data.set(key, storage)

        return result

    return wrapper


@cache_func_call
def get_embedding(text, model="text-embedding-ada-002"):
    openai.api_type = api_type
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key

    text = text.replace("\n", " ")
    try:
        return openai.Embedding.create(input=[text],
                                       engine=model)['data'][0]['embedding']
    except openai.error.RateLimitError as e:
        print(e)
        time_to_sleep = re.findall(r"retry after (\d+) second", str(e))[0]
        print(
            colored(
                f"(get_embedding) Rate limit exceeded. Waiting for "
                f"{time_to_sleep} seconds...", "yellow"))
        time.sleep(int(time_to_sleep))

        return get_embedding(text, model="text-embedding-ada-002")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_question_inbound(question, known_questions, threshold=0.8):

    v_q = get_embedding(question)
    for x in known_questions:
        sim = cosine_similarity(v_q, get_embedding(x))
        if sim > threshold:
            return True
    return False


# %%
def get_llm() -> object:
    openai.api_type = api_type
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key
    api = AzureGPTClient()
    return api


if __name__ == "__main__":
    api = get_llm()
    rst = api.reply(agent_name="user", msg="Hello")
    print(rst)
