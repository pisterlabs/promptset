import openai
from typing import List
# from utils.constants import OPENAI_API_KEY
from tqdm import tqdm
import time

OPENAI_API_KEYS = []
api_key_ptr = 0
openai.api_key = OPENAI_API_KEYS[api_key_ptr]

def request(
        prompt: str,
        engine='text-curie-001',
        max_tokens=64,
        temperature=1.0,
        top_p=1.0,
        n=2,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
):
    # retry request (handles connection errors, timeouts, and overloaded API)
    global api_key_ptr
    while True:
        try:
            # print(prompt)
            # print(max_tokens, temperature, top_p, n, stop, presence_penalty, frequency_penalty)
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            # print(response)
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            # time.sleep(1)
            api_key_ptr = (api_key_ptr+1) % len(OPENAI_API_KEYS)
            openai.api_key = OPENAI_API_KEYS[api_key_ptr]

    # print(response)
    generations = [gen['text'].strip() for gen in response['choices']]
    generations = [_ for _ in generations if _ != '']
    # print(generations)
    return generations
