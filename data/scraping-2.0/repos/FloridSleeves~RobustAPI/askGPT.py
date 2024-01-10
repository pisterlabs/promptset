import time
import fire
import os
import json
import numpy as np
import tqdm
import openai
from utils import generatePrompt, generateShot

def _call_gpt_api(
    model_name,
    prompt,
    keys,
    qid,
    temperature,
    top_p,
    n_generations
):
    response = None
    try_idx = 0
    while True:
        try:
            current_key = keys[try_idx % len(keys)]
            openai.api_key = current_key
            response = openai.ChatCompletion.create(
                model = model_name,
                messages = [
                    {"role": "user", "content": prompt}    
                ],
                temperature = temperature,
                top_p = top_p,
                n = n_generations
            )
            break
        except Exception as e:
            print('Retry:', qid)
            try_idx += 1
            time.sleep(10 + np.random.rand()*10)
    return response

def main(
    model_name: str,
    api_keys_file: str,
    result_dir: str,
    shot_number: int, # 0 for zero-shot, 1 for one-shot, -1 for irrelevant one-shot
    temperature: float = 0.2,
    top_p: float = 0.9,
    n_generations: int = 1,
    question: str = "./dataset/question.jsonl",
    shot_type: str = "example"
):
    start_time = time.time()
    os.makedirs(result_dir, exist_ok=True)

    print("Loading API keys...")
    with open(api_keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    
    print("Generating prompt samples...")
    samples = []
    with open(question, "r") as f:
        lines = f.readlines()
        prompts = [line.strip() for line in lines]
        for i, p in enumerate(prompts):
            check_path = os.path.join(result_dir, str(i) + ".json")
            if os.path.isfile(check_path):
                continue
            p = json.loads(p)
            shots = []
            try:
                shots = generateShot(p['api'], number = shot_number, type = shot_type)
            except:
                print("ERROR: No shots for api", p['api'])
                continue
            prompt = generatePrompt(p['api'], p['question'], shots, type = shot_type)
            samples.append((prompt, i, p['api']))
    print("Total samples:", len(samples))
       
    print("Generating responses...")
    for prompt, qid, api in tqdm.tqdm(samples):
        response = _call_gpt_api(
            model_name,
            prompt,
            keys,
            qid,
            temperature,
            top_p,
            n_generations
        )
        generation_dict = {'api': api, 
                           'prompt': prompt, 
                           'response': response['choices'][0]['message']['content'],
                           'gpt_response': response}
        
        fout = open(os.path.join(result_dir, str(qid) + ".json"), 'w')
        fout.write(json.dumps(generation_dict))
        fout.close()
    
    print(f"Elapsed time: {time.time() - start_time}")

if __name__ == "__main__":
    fire.Fire(main)
