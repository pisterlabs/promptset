import os
import openai
import tiktoken
from enum import IntEnum
from time import time

openai.api_key = os.getenv("OPENAI_API_KEY")

MODELS = [("text-ada-001", 0.0016), ("text-babbage-001", 0.0024),
          ("text-curie-001", 0.0120), ("text-davinci-003", 0.1200)]

session_total_cost = 0

class Model(IntEnum):
    ADA = 0
    BABBAGE = 1
    CURIE = 2
    DAVINCI = 3

def count_tokens(model, prompt, show_output=True):
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(prompt))
    if show_output:
        print(f"{tokens} tokens")
    return tokens


def predict(prompt, model=Model.DAVINCI, temp=0.5, max_tokens=100, top_p=1, freq_penalty=0.5, pres_penalty=0):
    global session_total_cost

    MODEL = MODELS[model][0]
    COST = MODELS[model][1] / 1000

    total_tokens = count_tokens(MODEL, prompt)
    print(f"MODEL: {MODEL:-^50}")
    print(f"PROMPT: {prompt}\n")

    time_start = time()
    response = openai.Completion.create(
        model=MODEL,
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=freq_penalty,
        presence_penalty=pres_penalty
    )
    total_time = time() - time_start

    response_text = response.choices[0].text
    total_tokens += count_tokens(MODEL, response_text)
    cost = total_tokens * COST
    session_total_cost += cost
    print(f"RESPONSE: {response_text}")
    print(f"TOTAL TOKENS: {total_tokens}")
    print(f"TOTAL COST: {cost} USD in {total_time:.2f} seconds")
    print(f"TOTAL COST SESSION: {session_total_cost} USD")
    return response_text

if __name__ == "__main__":
    predict("Hello, world!", Model.DAVINCI)
