"""
# Created: 2023-10-17 12:39
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Yi Yang & Qingwen Zhang

# Only this code is licensed under the terms of the MIT license. All other references are subjected to their own licenses.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
import openai
import os, json
from dotenv import load_dotenv, find_dotenv
import fire
from utils.prompt import *
from utils.mics import read_all_command, output_result, wandb_log, create_save_name, save_response_to_json
import time
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))

def get_completion_from_user_input(user_input, provide_detailed_explain=False, provide_few_shots = False, step_by_step = False, model="gpt-3.5-turbo", temperature=0, num_shots=4):
    fix_system_message = system_message
    if step_by_step:
        fix_system_message = step_system_message
        
    messages =  [  
    {'role':'system', 'content': fix_system_message},
    ]

    if not step_by_step and provide_detailed_explain:
        messages.append({'role':'assistant', 'content': f"{delimiter}{assistant}{delimiter}"})

    if provide_few_shots:
        if num_shots == 4:
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_2})
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_3}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_3})
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_4}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_4})
        elif num_shots == 3:
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_2})
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_3}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_3})
        elif num_shots == 2:
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_2}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_2})
        elif num_shots == 1:
            messages.append({'role':'user', 'content': f"{delimiter}{few_shot_user_1}{delimiter}"})
            messages.append({'role':'assistant', 'content': few_shot_assistant_1})

    messages.append({'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # control the randomness of the model's output
    )
    return response.choices[0].message["content"]

def main(
    csv_path: str = "llc/assets/ucu_subset.csv", #"code/llvm/data/ucu.csv",
    temperature: float = 0.0,
    provide_detailed_explain: bool = False,
    provide_few_shots: bool = False,
    step_by_step: bool = False,
    model: str = "gpt-3.5-turbo",
    debug_len: int = 10,
    slurm_job_id: str = "00000",
    resume: bool = False,
    start_from: str = "0",
    num_shots: int = 4,
):

    commands_w_id, tasks, gt_array = read_all_command(csv_path)
    print("1. Finished Read all commands!")
    model_name = create_save_name(model, provide_detailed_explain, provide_few_shots, step_by_step, debug_len, num_shots=num_shots)
    wandb_log(provide_detailed_explain, provide_few_shots, step_by_step, model_name, debug_len, slurm_job_id, num_shots=num_shots)

    all_results = []
    json_file_path = f"{BASE_DIR}/assets/result/{model_name}_{slurm_job_id}.json" # PLEASE DO NOT CHANGE THIS PATH
    numpy_file_path = f"{BASE_DIR}/assets/result/{model_name}_{slurm_job_id}.npy" # PLEASE DO NOT CHANGE THIS PATH
    os.makedirs(f"{BASE_DIR}/assets/result", exist_ok=True)

    if resume and os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            data = json.load(f)
        existing_ids = set([d['id'] for d in data])
        print(f"\n{bc.HEADER}Resume from {json_file_path}{bc.ENDC}, {len(all_results)} results loaded.")
        for d in data:
            all_results.append(d['response'])
    cnt = 0
    
    start_from = int(start_from)
    for (i, command) in commands_w_id:
        if resume and os.path.exists(json_file_path) and i in existing_ids:
            continue
        #if i< start_from*100 or i >= (start_from+1) * 100:
        #    continue

        start_time = time.time()

        for delay_secs in (2**x for x in range(0, 6)):
            try: 
                response = get_completion_from_user_input(command, 
                                                            provide_detailed_explain=provide_detailed_explain, provide_few_shots=provide_few_shots, step_by_step=step_by_step, \
                                                            model=model, temperature=temperature, num_shots=num_shots)

                style_response = {'id': i, 'command': command, 'response': response}
                save_response_to_json(style_response, json_file_path)

                if (i % 100 == 0 and debug_len == -1) or (debug_len>0):
                    print(f"\n===== command {bc.BOLD}{i}{bc.ENDC}: {command} =====================\n")
                    print(f"> {response}")

                all_results.append(response)
                cnt = cnt + 1
                break
        
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + randomness_collision_avoidance
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)
                continue
        if cnt>debug_len and debug_len != -1: # debugging now
            break

    output_result(numpy_file_path, json_file_path, model_name, all_results, gt_array, tasks, debug_len=debug_len)

if __name__ == "__main__":
    start_time = time.time()

    # read environment variables from .env
    _ = load_dotenv(find_dotenv())
    openai.api_key  = os.getenv('OPENAI_API_KEY')

    fire.Fire(main)
    print(f"Time used: {time.time() - start_time:.2f} s")