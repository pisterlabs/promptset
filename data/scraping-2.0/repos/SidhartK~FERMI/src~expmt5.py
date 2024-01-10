import argparse
from typing import Optional
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np

# torch
import torch

# init hugging face
from openai import OpenAI

from prompts.expmt5_prompts import QUERY_PROMPT, FEW_SHOT_PROMPTS, SYSTEM_PROMPT_STRING
from utils import accuracy_metric
from vec_store import Model, EvalDB

from unidecode import unidecode

import json

load_dotenv()

DATASET_PATH = "./data/verified_data/dataset.json"
OUTPUT_STORE_PATH = "./results/expmt4/output_store.json"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# def fp_score(answer, prediction):
#     log_diff = np.abs(np.log10(prediction) - np.log10(answer))
#     percentage_diff = log_diff/(np.abs(np.log10(answer)) + 1e-8)
#     return 1 - min(percentage_diff/2.0, 1.0)

class FermiProblem:
    def __init__(self, question, units, context, answer, question_type=None):
        self.question = question
        self.units = units
        self.answer = answer
        self.context = context
        self.question_type = question_type


class SamplePredictor(Model):
    def __init__(self, model_name="gpt-4", specialty="reasoning", temperature=0):
        self.model_name = model_name
        self.specialty = specialty
        self.model_type = model_name + "-" + specialty + "-" + temperature
        self.temperature = temperature
        self.few_shot_prompts = FEW_SHOT_PROMPTS[specialty]

    def chat(self, messages):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
        )
        return chat_completion.choices[0].message.content

    def print_metadata(self, metadata):
        print(f"Question: {metadata['question']} (units: {metadata['units']}); \nContext: {metadata['context']}\nCorrect Answer: {metadata['answer']}")
        print(f"Prediction is: {metadata['prediction']} \nQuestion Summary is: {metadata['summary']}\nProgram ({'valid' if metadata['program_valid'] else 'invalid'}):\n```python\n{metadata['program']}\n```")

    def get_query(self, fp: FermiProblem):
        return QUERY_PROMPT.format(question=fp.question, units=fp.units, context=fp.context)

    def evaluate(self, fp, verbose=False):
        metadata = {
            "question": fp.question,
            "units": fp.units,
            "context": fp.context,
            "question_type": fp.question_type,
            "answer": fp.answer,
        }

        messages = [{"role": "system", "content": SYSTEM_PROMPT_STRING}]
        for few_shot_prompt in self.few_shot_prompts:
            user_question = few_shot_prompt.split("User:")[1].split("Assistant:")[0].strip('\n')
            messages.append({"role": "user", "content": user_question})
            assistant_response = "\n" + few_shot_prompt.split("Assistant:")[1].split("User:")[0].strip('\n')
            messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": self.get_query(fp)})

        if verbose:
            print(f"\n{'-' * 110}\n{'-' * 45} RUNNING MODEL {'-' * 45}\n{'-' * 110}\n\n")

        preds = self.chat(messages)

        metadata = {
            **metadata,
            "llm_output": preds,
            "summary": "",
            "program": "",
            "program_valid": False,
            "prediction": 1.0,
        }
        accuracy = 0.0

        try:
            summary = preds.split("SUMMARY:=")[-1].split("PROGRAM:=")[0].strip('\n')

            program = preds.split("PROGRAM:=")[-1].strip('\n')
            program_lines = program.split("\n")
            program_lines = [line if line[0] != 'Q' else "# " + line for line in program_lines]
            program_lines = [line.replace(",", "") if line[0] != "#" else line for line in program_lines]
            program = "\n".join(program_lines)

            loc = {}
            exec(program, globals(), loc)
            prediction = loc["A0"]

            accuracy = accuracy_metric(fp.answer, prediction)
            metadata = {
                    **metadata,
                    "summary": summary,
                    "program": program,
                    "program_valid": True,
                    "prediction": prediction,
                }

        except:
            verbose = True

        return {
            "performance": accuracy,
            "evaluate_metadata": json.dumps(metadata),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--model', type=str, default="gpt-4", help="OpenAI model to use")
    parser.add_argument('--temp', type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help="Path to the dataset")
    parser.add_argument('--output_store', type=str, default=OUTPUT_STORE_PATH, help="Path to the output store")
    parser.add_argument('-N', type=int, default=-1, help="Number of test points")
    parser.add_argument('--save_each', action="store_true", default=True, help="Path to the output store")
    parser.add_argument('--verbose', action='store_true', help='Should the model be verbose?')
    args = parser.parse_args()
    np.random.seed(args.seed)

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    predictor = SamplePredictor(model_name=args.model, temperature=args.temp)

    acc, pars, output_store = predictor.run(
                                        dataset,
                                        N=args.N if args.N > 0 else None,
                                        output_store_path=args.output_store,
                                        verbose=args.verbose
                                    )

    print("Average Parsable Percentage: ", pars)
    print("Average Accuracy: ", acc)

    with open(args.output_store, 'w') as f:
        json.dump(output_store, f)
