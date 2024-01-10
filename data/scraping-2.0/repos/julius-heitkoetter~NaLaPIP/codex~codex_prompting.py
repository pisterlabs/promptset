from openai import OpenAI
import os
import pandas as pd 
import argparse
from tqdm import tqdm
import json

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.0
RANDOM_SEED = 123
NUM_TRIES = 5

#Directories to get few-shot prompting from
DIR_TRAINING_DATA = "codex/training_data"
FILE_PREDICATES = os.path.join(DIR_TRAINING_DATA, "predicates.js")
FILE_STIMULI = os.path.join(DIR_TRAINING_DATA, "stimuli.csv")

#Directories to get input files from
DIR_DATA = "data"

#Directories to save files to
DIR_EXPERIMENTS = "experiments"
DIR_CODEX = "codex"
DIR_COMPLETIONS = "completions"
DIR_PROMPTS = "prompts"

#Prompting skeleton
SYSTEM_INTEL = "You are OpenAI's GPT model, answer my questions about converting english questions to javascript conditionals as correctly as you can."
PROMPT_HEADER= """
You will output code to a probabilistic programming language built in java script. The code you output will be a conditional for a world that starts with 8 purple and blue boxes, stacked on top of each other, on a platform. Please only output a single line of code, nothing else.

The following are the predicates you may use. You can only use these predicates, do not invent other predicates.
"""
PROMPT_MIDDLE="""
Below are some examples 
"""
EXAMPLES_TEMPLATE= """
English: {english_question}
Code: {code_conditional}
"""
PROMPT_QUESTION="""
English: {english_question}
Code: 
"""

def construct_prompt(df_query, task_id):
    prompt = PROMPT_HEADER
    with open(FILE_PREDICATES, "r") as f:
        prompt += f.read()
    prompt += PROMPT_MIDDLE

    df_examples = pd.read_csv(FILE_STIMULI, index_col="task_id", keep_default_na=False)
    
    for _, row in df_examples.iterrows():
        example_header = EXAMPLES_TEMPLATE.format(
            english_question=row["english_question"],
            code_conditional=row["code_conditional"]
            )
        prompt += example_header
    
    prompt += PROMPT_QUESTION.format(english_question=df_query.loc[task_id, 'english_question'])
        
    return prompt

def query_llm(prompt):

    i = 0
    completion = None
    client = OpenAI()
    while i < NUM_TRIES:
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_INTEL},
                                                {"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            i = NUM_TRIES #Break the loop if completion sucseeded
        except Exception as e:
            print(e)
            print("WARNING: Failed to querey OpenAI. Trying again")
        i+= 1

    if completion is None:
        raise ValueError("ERROR: Not able to query openai sucessfully")
    
    return completion.choices[0].message.content.strip()

def parse(llm_output):

    parsed_output = llm_output

    return parsed_output

def run_experiment(input_file_name: str, experiment_id: str = None):
    experiment_id = experiment_id or datetime.datetime.now().strftime('run-%Y-%m-%d-%H-%M-%S')
    ckpt_dir = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_CODEX)
    os.makedirs(os.path.join(ckpt_dir, DIR_PROMPTS), exist_ok=True)

    df = pd.read_csv(
        os.path.join(DIR_DATA, input_file_name),
        index_col="task_id",
        keep_default_na=False,
    )

    completions = []
    results = []
    # Query OpenAI for completions
    for task_id in tqdm(df.index):
            
        prompt_text = construct_prompt(df, task_id)
        completion = query_llm(prompt_text)
        completions.append(completion)
            
        with open(os.path.join(ckpt_dir, DIR_PROMPTS, f"prompt_task_{task_id:03d}.json"), "w") as f:
            prompt_json = {
                "task_id": task_id,
                "prompt_text": prompt_text,
            }
            json.dump(prompt_json, f)

        # Parse completions data
        d = {
            "task_id": task_id,
            "codex_conditional_str": parse(completion),
        }
        results.append(d)

    results_json = {"results": results}
    
    df_results = pd.DataFrame(results_json["results"]).fillna('')
    df_results = df_results.set_index("task_id")
    
    df_results = df.join(df_results)
    df_results.to_csv(os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_CODEX, "results.csv"))
    
    return df_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_name")
    parser.add_argument("experiment_id")

    args = parser.parse_args()

    run_experiment(
        input_file_name = args.input_file_name,
        experiment_id = args.experiment_id,
    )

if __name__ == "__main__":
    main()

