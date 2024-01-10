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
async def get_completion(story, query, genders, session, semaphore, pbar):
    async with semaphore:
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
                "model": "gpt-4-1106-preview",
                "seed": 42,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Your task is to write Pyro code to model relationships in stories, considering gender information, and answer queries about them. Please consider a full list of relationships include those of siblings, in-laws (daughter-in law, son in-law, mother-in law, aunt in law, uncle in-law, etc.) so look beyon just the story and consider all familial relationships in your code. "},
		    {"role": "system", "content": """
                    here is an example of pyro code for the following story: \n \n 
                    [Darnell] loved his mother, [Theresa]. [Theresa] was so proud of her daughter [Amanda] for getting straight A's this semester. [Darnell] is going to the arcade with his sister, [Michelle]. [Michelle] was excited for today, its her daughter's, [Theresa], spring break. She will finally get to see her. \n \n \n
                    and for this query: \n \n ('Amanda', 'Michelle') \n \n which had a target of "sister". \n \n
                    Please ensure that the output of the answer is of the relational form e.g. "'mother', 'daughter', 'sister', 'aunt', 'cousin', 'grandmother', 'granddaughter'": \n \n this was the code:
                    import torch
                    import pyro
                    import pyro.distributions as dist

                    # use the provided genders of the individuals
                    genders = {'Amanda': 'female', 'Theresa': 'female', 'Michelle': 'female', 'Darnell': 'male'}

                    # Define a simple family tree model in Pyro
                    def family_tree_model():
                        # Define the relationships and their initial probabilities
                        relationships = ['mother', 'daughter', 'sister', 'other']
                        rel_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])  # Equal probabilities
                        
                        # Theresa is the mother of Amanda and Michelle; Darnell is the brother of Michelle.
                        # We reflect these relationships in our model
                        # For simplicity, we use indices: mother=0, daughter=1, sister=2, etc. please write out conditional indices of all possibilities.
                        # Theresa -> Amanda (mother)
                        pyro.sample('Theresa_Amanda', dist.Categorical(probs=torch.tensor([1.0, 0.0, 0.0, 0.0])))
                        # Theresa -> Michelle (mother)
                        pyro.sample('Theresa_Michelle', dist.Categorical(probs=torch.tensor([1.0, 0.0, 0.0, 0.0])))
                        # Darnell -> Michelle (sister)
                        pyro.sample('Darnell_Michelle', dist.Categorical(probs=torch.tensor([0.0, 0.0, 1.0, 0.0])))
                        
                        # Inference for Amanda's relationship to Michelle
                        # Since Theresa is the mother of both Amanda and Michelle, Amanda and Michelle are sisters
                        amanda_michelle_rel = pyro.sample('Amanda_Michelle', dist.Categorical(probs=torch.tensor([0.0, 0.0, 1.0, 0.0])))
                        
                        return amanda_michelle_rel.item()

                    # Run the model to infer the relationship between Amanda and Michelle
                    most_likely_relationship = family_tree_model()
                    relationship = relationships[most_likely_relationship]

                    print(f"The inferred relationship between Amanda and Michelle is: {relationship}")


                      """},
                    {"role": "user", "content": "use the following steps to solve the question in the next prompt: First consider all of the conditionals provided from the story and then write those out in pyro like the example above.Think of the correct relationship and please ensure that you consider all types of relationships like mother-in-law, sister-in-law, uncle-in-law, brother-in-law, etc." },
                    {"role": "user", "content": f"Story: {story} \nGenders: {genders} \nQuery: {query}. "}
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

async def generate_pyro_code(stories, queries, genders, max_parallel_calls, timeout):
    semaphore = asyncio.Semaphore(value=max_parallel_calls)
    pbar = tqdm(total=len(stories))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            tasks = []
            for story, query, gender in zip(stories, queries, genders):
                task = get_completion(story, query, gender, session, semaphore, pbar)
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pbar.close()
    return results

# Load the CSV data
file_path = 'data_emnlp_final/data_06b8f2a1/2.2,2.3_train.csv'
df = pd.read_csv(file_path)
df = df.iloc[:200]

# Prepare stories, queries, and gender information
stories = df['clean_story'].tolist()
queries = df['query'].tolist()
genders = df['genders'].tolist()

# Run the asynchronous code generation
max_parallel_calls = 25
timeout = 60
loop = asyncio.get_event_loop()
pyro_code_results = loop.run_until_complete(generate_pyro_code(stories, queries, genders, max_parallel_calls, timeout))

# Process results to get Pyro code
pyro_code_snippets = [{
    "story": story,
    "query": query,
    "genders": gender,
    "pyro_code": result
} for story, query, gender, result in zip(stories, queries, genders, pyro_code_results)]

# Save or process pyro_code_snippets as needed
output_file_path = 'babi_pyro_code_results_with_gender_GPT4_turbo_v5.json'
with open(output_file_path, 'w') as out_file:
    json.dump(pyro_code_snippets, out_file, ensure_ascii=False, indent=2)





