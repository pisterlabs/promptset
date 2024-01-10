#!/usr/bin/env python

import os
from datetime import datetime
import glob
import sys
from concurrent.futures import ThreadPoolExecutor
import time
import itertools
import pandas as pd
import pickle

# Set up the open ai input
import openai
openai_client = openai.OpenAI(api_key = os.environ['OPENAI_API_KEY'])

# for xetcache.  Note this gives a lot of too many open file errors.
import xetcache
xetcache.set_xet_project("LLM_evolution", private=False)

# For diskcache.  This is a local cache but finer grained than the xetcache above.
import diskcache
cache = diskcache.Cache("./cache")
    
# Runs through a small subset of things to test it all out.
test_mode = True

# Change this to refresh results.  
date_key = "2023-12-12"

#############################

def answer_in_backticks(xd):
    """
    Extracts the answer from between triple backticks like ```
    """
    
    text = xd["response"]

    first_backtick = text.find('```')
    second_backtick = text.find('```', first_backtick + 3)

    if first_backtick != -1 and second_backtick != -1:
        return text[first_backtick + 3:second_backtick]
    else:
        return text  # or handle the case where there aren't two backticks


def evaluation_multiple_choice_match(xd):
    """
    Gives a score of 1 if the choices match and 0 otherwise.
    """

    known_answer = xd["known_answer"]
    answer = xd["answer"]

    # Modify this for sure
    if known_answer is not None and answer.startswith(known_answer):
        score = 1.
    else:
        score = 0.

    return score




##############################
# Data loaders

class MMLU: 

    def load(self):
        """
        Load the data.  Will return a list of lists.
        """

        output = []

        if test_mode:
            path = "data/mmlu/test/management_test.csv"
        else:
            path = "data/mmlu/test/*.csv"

        for filename in glob.glob(path):
            data = pd.read_csv(filename)
            output.append(list(self.formulate_prompt(row) for row in data.values))
     
        return output

    def formulate_prompt(self, input_row):
     
        question = input_row[0]
        a, b, c, d = input_row[1], input_row[2], input_row[3], input_row[4]
        answer = input_row[5]

        prompt = f"""In the following multiple choice question, please state the answer as A, B, C, or D.
        {question}
        A. {a}
        B. {b}
        C. {c}
        D. {d}
        """

        return {"base_prompt" : prompt, 
                "known_answer" : answer, 
                "source_data" : input_row, 
                "evaluation_method" : "evaluation_multiple_choice_match",
                "source_tag" : self.__class__.__name__}


all_datasets = [MMLU()]

###############################
# Prompt engineering techniques


class PromptPassThrough:

    tag = "unmodified"
    answer_extraction_method = "answer_in_backticks"
    
    @staticmethod 
    def modify_prompt(text):
        return f"""{text}
        Please enclose your answer in triple backticks (for example: ```XYZ```, where XYZ is your answer).
        """ 


class PromptSimpleExplanationFollowing:
    
    tag = "simple_explanation_following"
    answer_extraction_method = "answer_in_backticks"

    def tag(self): 
        return "simple_explanation_following"
    
    @staticmethod 
    def modify_prompt(text):
        return f"""
        {text}
        Please enclose your answer in triple backticks (for example: ```XYZ```, where XYZ is your answer), and then 
        follow the answer with a simple explanation of your reasoning.
        """


class PromptSimpleExplanationLeading:
    tag = "simple_explanation_following"
    answer_extraction_method = "answer_in_backticks"

    @staticmethod 
    def modify_prompt(text):
        return f"""
        {text}
        Please provide a step-by-step explanation of your reasoning, then enclose your final answer 
        in triple backticks (for example: ```XYZ```, where XYZ is your answer).
        """


all_modifications = [
    PromptPassThrough,
    PromptSimpleExplanationFollowing, 
    PromptSimpleExplanationLeading]


##########################
# Models

all_models = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-0301"]# , "gpt-4-turbo", "gpt-4-1106-preview"]

##########################
# runner.

completion_count = 0
total_count = 0

def report_task_completed():
    global completion_count
    global total_count

    completion_count += 1
            
    sys.stderr.write('.')
    if completion_count % 100 == 0: 
        sys.stderr.write(f" {completion_count} / {total_count}\n")

@cache.memoize(typed=True, expire = None, tag="run_prompt_list")
def load_or_run_prompt_list(input_list):
    # Used with the decorator above
    return run_prompt_list(input_list)    

@xetcache.xetmemo
def run_prompt_list(input_list):
    """
    Runs a distinct collection of prompts that gets cached remotely. 
    """
    
    with ThreadPoolExecutor(max_workers=16) as executor:

        local_futures = [None]*len(input_list)

        def compute_result(xd):
            response_full, response = query_chatgpt(xd["prompt"], xd["model"], xd.get("parameters", {}))

            output_d = xd.copy()
            output_d["response"] = response
            output_d["response_full"] = response_full 

            answer_extraction_name = xd.get("answer_extraction_method", None)

            if answer_extraction_name:
                answer_extraction = globals()[answer_extraction_name]
                output_d["answer"] = answer_extraction(output_d)
            else:
                output_d["answer"] = output_d["response"] 

            evaluation_method_name = xd.get("evaluation_method", None)
            if evaluation_method_name:
                evaluation_method = globals()[evaluation_method_name]
                output_d["score"] = evaluation_method(output_d) 

            return output_d

        for i, xd in enumerate(input_list):
            # xd here is the input dictionary of queries 
            local_futures[i] = executor.submit(compute_result, xd)
        
        ret = [None]*len(local_futures)

        for i, f in enumerate(local_futures):
            ret[i] = f.result()
            report_task_completed()
    
    return ret

seen_exceptions = set()


# Cache local low-level results in the disk-backed cache 
@cache.memoize(typed=True, expire = None, tag="query_chatgpt")
def query_chatgpt(prompt, model, parameters):

    backoff_time = 1.0
    n_tries = 0

    # Make the API call
    while True:
        try:
            response = openai_client.chat.completions.create(
                model = model,
                messages = [{"role" : "system", 
                            "content" : "You are a helpful assistant."}, 
                            {"role" : "user", "content" : prompt}]
            )
            break
        except openai.RateLimitError:

            if n_tries <= 16:
                time.sleep(backoff_time)
                backoff_time *= 2.0
                n_tries += 1
                sys.stderr.write('R')
                continue
            else:
                raise
        
        except Exception as e:
            if n_tries <= 4:
                n_tries += 1
                time.sleep(8.)
                sys.stderr.write('E')
                continue
            else:
                error_str = str(e)
                global seen_exceptions
                if error_str not in seen_exceptions:
                    sys.stderr.write(f"\n\nUnrecoverable exception encountered: {e}\n")
                    sys.stderr.write(f"    Model = {model}, Context = {prompt}, \n\n")
                    seen_exceptions.add(error_str)

                raise




    # Return the text part of the response
    return {"model_dump" : response.model_dump_json(), 
              "response" : response.choices[0].message.content}


def run_all():
    """
    Runs all the results in the current configuration.
    """

    # Due to xetcache and others, using nested executors will 
    with ThreadPoolExecutor(max_workers=4) as executor:

        output_futures = []
        n_prompt_variants = 0

        for d in (all_datasets[:1] if test_mode else all_datasets):
            base_prompt_lists = d.load()
                
            for (model, modification, base_prompts) in itertools.product(all_models, all_modifications, base_prompt_lists):

                # Get all the base prompts together here.
                prompts = [None]*len(base_prompts)
                for i, bp in enumerate(base_prompts):
                    p = bp.copy()
                    p["prompt"] = modification.modify_prompt(p["base_prompt"])
                    p["modification_tag"] = modification.tag
                    p["answer_extraction_method"] = modification.answer_extraction_method
                    p["model"] = model
                    p["date_key"] = date_key
                    prompts[i] = p
                        
                if test_mode:
                    prompts = prompts[:1]

                n_prompt_variants += prompts.len()
                output_futures.append(executor.submit(load_or_run_prompt_list, prompts))
    
        sys.stderr.write(f"Querying or running on {n_prompt_variants} prompt variants .\n")
        
        output_data = []
        for f in output_futures:
            output_data += f.result() 

    sys.stderr.write('\nCompleted.\n')

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"output_data/output_data-{timestamp}.pkl"
    pickle.dump(output_data, open(filename, "wb"))

    print(f"Output saved to {filename}")


if __name__ == '__main__':
    run_all()
                    
    













