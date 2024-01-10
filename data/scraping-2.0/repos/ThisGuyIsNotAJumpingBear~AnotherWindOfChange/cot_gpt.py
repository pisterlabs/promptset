from openai import OpenAI
from dotenv import load_dotenv
from multiprocessing import Pool
import os
import time
import pickle
from tqdm import tqdm
import numpy as np
from utils import load_instances, load_labels

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], organization="org-6u1yKGMuXAyb3dStdjvmFHMo")

def prompt_gpt(prompt_obj):
    """
    chatCompletion does not allow batched prompts unfortuantely.
    """

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": f"{prompt_obj['prompt']}"},
    ],
    temperature=0.1,
    n=1,
    frequency_penalty=1.5,
    max_tokens=300)
    # have to sleep to make sure openai limits are not reached
    time.sleep(2)
    return {"id": prompt_obj["id"], "response": response.choices[0].message.content, "label": prompt_obj["label"]}

def create_zero_shot_cot_prompt(data_entry):
    #following zero-shot cot paper
    tweet1 = data_entry["tweet1"]["text"]
    tweet2 = data_entry["tweet2"]["text"]
    word = data_entry["word"]
    date1 = data_entry["tweet1"]["date"]
    date2 =  data_entry['tweet2']["date"]


    query = f"Tweet-1: {tweet1} Date: {date1} Tweet-2: {tweet2} Date: {date2} Question: Is the meaning of {word} different in the last 2 tweets?"

    return f'''Q: {query} A: Let's think step by step.'''

def prep_dataset():
    gpt_prompts = []
    labels = load_labels()
    
    instances = load_instances()
    for data in instances:
        if data["id"] in labels.keys():
            gpt_prompts.append({'id': data["id"], "prompt":create_zero_shot_cot_prompt(data), "label": labels[data["id"]]})
    
    return gpt_prompts

def parallel_prompt_gpt(prompts, start, end):
    core_prompts = prompts[start:end]

    chatgpt_answers = []
    pbar = tqdm(total=len(core_prompts))
    for prompt in core_prompts:
        chatgpt_answers.append(prompt_gpt(prompt))
        pbar.update(1)
    
    return chatgpt_answers

def main():
    num_procs = 4
    prompts = prep_dataset()

    num_samples = 100

    idx_arr = np.random.choice(np.arange(len(prompts)), size=num_samples, replace=False)
    scaled_down_prompts = []

    for idx in idx_arr:
        scaled_down_prompts.append(prompts[idx])
    
    prompts = scaled_down_prompts

    partition = len(prompts) // num_procs

    pooling_partition_arr = []
    for i in range(num_procs):
        if i == num_procs-1:
            pooling_partition_arr.append((prompts, i * partition, len(prompts)))
        else:
            pooling_partition_arr.append((prompts, i * partition, (i+1) * partition))

        
    pool = Pool(num_procs)
    results = pool.starmap(parallel_prompt_gpt, pooling_partition_arr)
    pool.close()
    pool.join()

    with open('data/gpt3.5_cot.pkl', 'wb') as fp:
        pickle.dump(results, fp)
    
    with open('data/gpt3.5_cot.pkl', 'rb') as fp:
        chatgpt_answers = pickle.load(fp)
        print(chatgpt_answers)

def parse_cot():
    with open('data/gpt3.5_cot.pkl', 'rb') as fp:
        chatgpt_answers = pickle.load(fp)

    f = open('cot_responses.txt', 'w')

    for core_result in chatgpt_answers:
        for answer in core_result:
            f.write(f"{answer['response']} {answer['label']} \n")
            f.write("-" * 80)
            f.write("\n")
    f.close()  

if __name__ == "__main__":
    parse_cot()