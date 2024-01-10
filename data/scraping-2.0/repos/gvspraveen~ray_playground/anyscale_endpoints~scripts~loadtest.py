import openai
import os 
api_key = os.getenv('ANYSCALE_API_KEY')
api_base = "https://api.endpoints.anyscale.com/v1"

model = "meta-llama/Llama-2-7b-chat-hf"
fine_tuned_model = "meta-llama/Llama-2-13b-chat-hf"


def get_response(model):
    try:
        response = openai.ChatCompletion.create(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won Australian open 2012 final and how many sets were played?"}],
            temperature=0.9,
            max_tokens=1200
        )
        # choice = response["choices"][0]
        # message = choice["message"]
        # content = message["content"]
        return response
    except Exception as e:
        return e.message


import time


TARGET_QPS = 1
last_fetched = 0

while True:
    now = time.monotonic()
    if now - last_fetched > (1 / TARGET_QPS):
        resp = get_response(model)
        resp1 = get_response(fine_tuned_model)
        print(resp)
        print(resp1)
        last_fetched = now 
    after = time.monotonic()
    time.sleep(max(0, 0.05 - (after - now)))
