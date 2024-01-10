import asyncio
import openai
import time
import json
from termcolor import colored
import os

from lists import general_purposes

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

'''!!! I HAVEN'T TESTED ALL THE FUNCTIONS RETURNED FROM GPT, BE CAREFUL AND USE GOOD JUGDEMENT BEFORE USING ANY OF THE FUNCTIONS!!!'''

'''CHECK FOR ENV VARIABLE FOR OPENAI API KEY, IF NOT SET ONE'''

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY here'


# TESTING IT OUT WITH 10 PURPOSES
purposes = general_purposes[:10]



def save_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def python_function_generator_async(purpose):
    global total_tokens_used

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful python programmer who writes excellent functions given a purpose for a function."},
            {"role": "user", "content": f""" write a single python function for the purpose: {purpose}"""}
        ]
    )

    tokens = response["usage"]["total_tokens"]


    response = response['choices'][0]['message']['content'] 

    return {'purpose': purpose, 'code': response, 'tokens_used': tokens}


# this is to make all the async calls at once
async def make_async_calls_full():
    tasks = []
    for purpose in purposes:
        
        print("async calls started""")
        tasks.append(loop.create_task(python_function_generator_async(purpose)))
    results = await asyncio.gather(*tasks)
    print("async calls finished""")
    save_to_json(results, 'async.json')


# this is to make async calls in two batches
async def make_async_calls():
    tasks = []
    half_length = len(purposes) // 2
    results = []  # To store results from all tasks
    for i, purpose in enumerate(purposes):
        print("Async calls started.")
        tasks.append(loop.create_task(python_function_generator_async(purpose)))

        # If we have created tasks for half of the purposes, wait for them to complete,
        # sleep for 60 seconds, and then continue.
        if i == half_length - 1:
            results.extend(await asyncio.gather(*tasks))  # Store the results from the first half
            print("First half of async calls finished.")
            tasks = []  # Clear the tasks list for the next half.
            await asyncio.sleep(60)

    # Await the remaining tasks and store their results.
    results.extend(await asyncio.gather(*tasks))
    print("Async calls finished.")
    save_to_json(results, 'async.json')

start_time = time.time()

loop = asyncio.get_event_loop()
loop.run_until_complete(make_async_calls())


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time elapsed: {elapsed_time} seconds")

total_tokens_used = 0

try:
    with open("async.json", "r") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(colored(f"Error decoding JSON: {e}", "red"))
else:
    for item in data:
        for key, value in item.items():
            if key == "tokens_used":
                print(colored(f"Tokens used for {item['purpose']}: {value}", "green"))
                total_tokens_used += value

print(colored(f"Total tokens used: {total_tokens_used}", "green"))

