import os
import argparse
import time
import json
import time
from tqdm import tqdm
import openai
import domain_utils
from domain_utils import *

MAX_GPT_RESPONSE_LENGTH = 100
MAX_BACKPROMPT_LENGTH = 15
STOP_PHRASE = "Verifier confirmed success" # what the verifier has to say
STOP_STATEMENT = "[ANSWER END]" # what we look for to end LLM response generation

def get_responses(engine, domain_name, specified_instances = [], run_till_completion=False, ignore_existing=False, verbose=False, multiprompting="", problem_type="", multiprompt_num=15, temp=0):
    domain = domain_utils.domains[domain_name]

    # Check for/set up relevant directories
    instances_dir = f"data/{domain_name}/"
    prompt_dir = f"prompts/{domain_name}/"
    output_dir = f"responses/{domain_name}/{engine}/"
    if multiprompting: output_dir += f"backprompting-{multiprompting}{f'-temp{temp}' if temp else ''}/"
    if problem_type: output_dir+=f"{problem_type}/"
    os.makedirs(output_dir, exist_ok=True)
    output_json = output_dir+"responses.json"
    
    # Load prompts
    with open(prompt_dir+f"prompts{f'-{problem_type}' if problem_type else ''}.json", 'r') as file:
        input = json.load(file)

    # Constrain work to only specified instances if flagged to do so
    if len(specified_instances) > 0:
        input = {str(x) : input[str(x)] for x in specified_instances}

    # Load previously done work
    output = {}
    if os.path.exists(output_json) and not ignore_existing:
        with open(output_json, 'r') as file:
            # NOTE: the following json file should be a dictionary of dictionaries (each representing an instance), each containing three things: prompts (an ordered list of prompts), responses (an ordered list of corresponding responses), and stopped (a boolean of whether generation has been stopped on purpose yet)
            output = json.load(file)
    
    # Loop over instances until done
    while True:
        failed_instances = []
        for instance in tqdm(input):
            instance_output = {"prompts":[input[str(instance)]], "responses":[], "stopped":False}
            finished_prompts = 0

            # Load actual instance data
            instance_location = f"{instances_dir}/instance-{instance}{domain.file_ending()}"
            try:
                with open(instance_location,"r") as fp:
                    instance_text = fp.read()
            except FileNotFoundError:
                print(f"{instance_location} not found. Skipping instance {instance} entirely.")
                continue

            # Check if this instance is already complete
            if instance in output:
                instance_output = output[instance]
                finished_prompts = len(instance_output["responses"])
                if instance_output["stopped"]:
                    if verbose: print(f"===Instance {instance} already completed===")
                    continue
                elif finished_prompts >= multiprompt_num:
                    if verbose: print(f"===Instance {instance} already maxed out at {finished_prompts} out of {multiprompt_num} backprompts===")
                    continue
                elif verbose: print(f"===Instance {instance} not complete yet. Continuing from backprompt {finished_prompts+1}===")
            elif verbose: print(f"===Instance {instance} has never been seen before===")

            # Loop over the multiprompts until verifier stops or times out
            while len(instance_output["responses"])< multiprompt_num and not instance_output["stopped"]:
                if len(instance_output["prompts"]) > len(instance_output["responses"]):
                    if verbose: print(f"==Sending prompt {len(instance_output['prompts'])} to LLM for instance {instance}: ==\n{instance_output['prompts'][-1]}")
                    llm_response = send_query(instance_output["prompts"][-1], engine, MAX_GPT_RESPONSE_LENGTH, stop_statement=STOP_STATEMENT, temp=temp)
                    if not llm_response:
                        failed_instances.append(instance)
                        print(f"==Failed instance: {instance}==")
                        continue
                    if verbose: print(f"==LLM response: ==\n{llm_response}")
                    instance_output["responses"].append(llm_response)
                if len(instance_output["prompts"]) == len(instance_output["responses"]):
                    backprompt_query = domain.backprompt(instance_text, instance_output, multiprompting)
                    instance_output["prompts"].append(backprompt_query)
                    if check_backprompt(backprompt_query):
                        instance_output["stopped"] = True
                        if verbose: print(f"==Stopping instance {instance} after {len(instance_output['responses'])} responses.==")
                output[instance]=instance_output
                with open(output_json, 'w') as file:
                    json.dump(output, file, indent=4)
        
        # Run till completion implementation
        if run_till_completion:
            if len(failed_instances) == 0:
                break
            else:
                print(f"Retrying failed instances: {failed_instances}")
                input = {str(x):input[str(x)] for x in failed_instances}
                time.sleep(5)
        else:
            print(f"Failed instances: {failed_instances}")
            break

def check_backprompt(backprompt):
    return STOP_PHRASE.lower() in backprompt.lower()

def send_query(query, engine, max_tokens, stop_statement="[ANSWER END]", temp=0, top_num=5):
    max_token_err_flag = False
    if '_chat' in engine:
        eng = engine.split('_')[0]
        # print('chatmodels', eng)
        messages=[
        {"role": "system", "content": "You are a constraint satisfaction solver that solves various CSP problems."},
        {"role": "user", "content": query}
        ]
        try:
            response = openai.ChatCompletion.create(model=eng, messages=messages, temperature=temp, stop=stop_statement)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT query execution: {}".format(e))
        text_response = response['choices'][0]['message']['content'] if not max_token_err_flag else ""
        return text_response.strip()        
    else:
        try:
            response = openai.Completion.create(
                model=engine,
                prompt=query,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_statement)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT query execution: {}".format(e))
        text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
        return text_response.strip()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        ')
    parser.add_argument('-d', '--domain', type=str, required=True, help='Problem domain to query for')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-r', '--run_till_completion', type=str, default="False", help='Run till completion')
    parser.add_argument('-s', '--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('-i', '--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('-b', '--backprompt', type=str, default='', help='If multiprompting, provide the type of backprompt to pass to the domain. Common types: zero, passfail, full, llm, top')
    parser.add_argument('-B', '--backprompt_num', type=int, default=15, help='If multiprompting, provide the maximum number of prompts to try. Double this number to get expected behavior for LLM backprompting')
    parser.add_argument('-n', '--end_number', type=int, default=0, help='For running from instance m to n')
    parser.add_argument('-m', '--start_number', type=int, default=1, help='For running from instance m to n. You must specify -n for this to work')
    parser.add_argument('-p', '--problem', type=str, default='', help='If doing a domain subproblem, specify it here')
    parser.add_argument('-t', '--temperature', type=float, default=0, help='Temperature from 0.0 to 2.0')
    args = parser.parse_args()
    engine = args.engine
    domain_name = args.domain
    if domain_name not in domain_utils.domains:
        raise ValueError(f"Domain name must be an element of {list(domain_utils.domains)}.")
    specified_instances = args.specific_instances
    verbose = args.verbose
    backprompt = args.backprompt
    backprompt_num = args.backprompt_num
    run_till_completion = eval(args.run_till_completion)
    ignore_existing = args.ignore_existing
    end_number = args.end_number
    start_number = args.start_number
    problem_type = args.problem
    temperature = args.temperature
    if end_number>0 and specified_instances:
        print("You can't use both -s and -n")
    elif end_number>0:
        specified_instances = list(range(start_number,end_number+1))
        print(f"Running instances from {start_number} to {end_number}")
    print(f"Engine: {engine}, Domain: {domain_name}, Verbose: {verbose}, Run till completion: {run_till_completion}, Multiprompt Type: {backprompt}, Problem Type: {problem_type}")
    get_responses(engine, domain_name, specified_instances, run_till_completion, ignore_existing, verbose, backprompt, problem_type, multiprompt_num=backprompt_num, temp=temperature)