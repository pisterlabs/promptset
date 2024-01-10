import openai
import pandas as pd
import json
import os
from openai import OpenAI
import re
from tqdm.asyncio import tqdm
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_random_exponential
import io
import contextlib

# Set the API key
os.environ["OPENAI_API_KEY"] = "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"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

class ProgressLog:
    def __init__(self, total):
        self.total = total
        self.done = 0

    def increment(self):
        self.done += 1

    def __repr__(self):
        return f"Done runs {self.done}/{self.total}."

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), retry_error_callback=lambda _: None)
async def get_completion(story, query, session, semaphore, pbar):
    async with semaphore:
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
                "model": "gpt-4",
                "seed": 42,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Your task is to write Pyro code to model relationships in stories and answer queries about them. Please consider all relational types"},
                    {"role": "system", "content": """Here is an example of story and complementary code you should attempt to write. For this given story: 

                    1 Lily is a frog.
                    2 Bernhard is a frog.
                    3 Bernhard is green.
                    4 Brian is a lion.
                    5 Brian is white.
                    6 Julius is a swan.
                    7 Julius is green.
                    8 Lily is green.
                    9 Greg is a swan.
                    10 What color is Greg?

                    to which the answer as green,

                    the corresponding pyro code looked like this: 

                    import pyro
                    import pyro.distributions as dist
                    import torch

                    # Example data (species to color mapping)
                    data = {
                        'frog': ['green', 'green'],
                        'lion': ['white'],
                        'swan': ['green', 'green', 'white']  # Assuming we have an observation that a swan can also be white
                    }

                    # Model
                    def species_color_model(data):
                        # Prior probabilities for each color for each species
                        species_colors = {
                            'frog': pyro.sample('frog_color', dist.Dirichlet(torch.tensor([1.0, 1.0]))),
                            'lion': pyro.sample('lion_color', dist.Dirichlet(torch.tensor([1.0, 1.0]))),
                            'swan': pyro.sample('swan_color', dist.Dirichlet(torch.tensor([1.0, 1.0])))
                        }

                        # Observations
                        for species, colors in data.items():
                            for color in colors:
                                if color == 'green':
                                    pyro.sample(f'obs_{species}_{color}', dist.Categorical(species_colors[species]), obs=torch.tensor(0))
                                elif color == 'white':
                                    pyro.sample(f'obs_species}_color}', dist.Categorical(species_colors[species]), obs=torch.tensor(1))

                    # Inference
                    conditioned_model = pyro.condition(species_color_model, data={})

                    # Assuming we want to infer the color of a 'swan'
                    # We would run inference here (e.g., using Pyro's MCMC or SVI) to get the posterior distribution of swan's color
                    
                     """},
                    {"role": "user", "content": f"Story: {story} \nQuery: {query}. "}
                ]
            }) as resp:
                resp.raise_for_status()
                response_json = await resp.json()
                result = response_json["choices"][0]['message']["content"]
        except aiohttp.ClientResponseError as e:
            print(f"HTTP Error: {e.status}")
            result = None
        except aiohttp.ClientError as e:
            print(f"Client Error: {e}")
            result = None
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            result = None
        except Exception as e:
            print(f"Unexpected error: {e}")
            result = None
        finally:
            pbar.update(1)
            return result

async def generate_pyro_code(stories, queries, max_parallel_calls, timeout):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    pbar = tqdm(total=len(stories))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            tasks = []
            for story, query in zip(stories, queries):
                task = get_completion(story, query, session, semaphore, pbar)
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pbar.close()
    return results

# Load the bAbI dataset
file_path = 'tasks_1-20_v1-2/en/qa16_basic-induction_test.txt'
with open(file_path) as f:
    lines = f.readlines()

# Group every 10 lines into a single story
stories = []
queries = []
for i in range(0, int(len(lines)/5), 10):
    story = "\n".join(lines[i:i+9]).strip()  # Story is the first 9 lines
    query = lines[i+9].strip()               # Query is the 10th line
    stories.append(story)
    queries.append(query)

# Run the asynchronous code generation
max_parallel_calls = 25
timeout = 60
loop = asyncio.get_event_loop()
pyro_code_results = loop.run_until_complete(generate_pyro_code(stories, queries, max_parallel_calls, timeout))

# Process results to get Pyro code
pyro_code_snippets = [{
    "story": story,
    "query": query,
    "pyro_code": result
} for story, query, result in zip(stories, queries, pyro_code_results)]

# Save or process pyro_code_snippets as needed
output_file_path = 'babi_q16_pyro_code_results_with_GPT4.json'
with open(output_file_path, 'w') as out_file:
    json.dump(pyro_code_snippets, out_file, ensure_ascii=False, indent=2)
