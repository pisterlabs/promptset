import argparse
import time
from typing import Optional
from dotenv import load_dotenv
import os
from tqdm import tqdm
import numpy as np

# torch
import torch

# init hugging face
from openai import OpenAI

from prompts.expmt0_prompts import QUERY_PROMPT, ASSISTANT_RESP_PROMPT, SYSTEM_PROMPT_STRING

from unidecode import unidecode

from utils import accuracy_metric, compile_fp, convert_units
import json

load_dotenv()

FEWSHOT_DATASET_PATH = "./data/verified_data/dataset-val.json"
DATASET_PATH = "./data/verified_data/dataset-test.json"
OUTPUT_STORE_DIR= "./results/final_results/expmt0.0/"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class SamplePredictor:
    def __init__(self, few_shot_prompts, model_name="gpt-4", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
        self.few_shot_prompts = few_shot_prompts

    @staticmethod
    def split_context_program(split):
        program = []
        context = []
        for segment in split[1:]:
            context_track = segment[0] == 'F'
            if context_track:
                context.append(segment)
            else:
                program.append(segment)
        program = '='.join(program[1:])
        context = 'CONTEXT:='+'='.join(context)
        answer = split[0]
        return answer, program, context

    def chat(self, messages):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
        )
        return chat_completion.choices[0].message.content

    def print_metadata(self, metadata):
        print(f"Question: {metadata['question']} (units: {metadata['units']}); \nContext: {metadata['context']}\nCorrect Answer: {metadata['answer']}")
        print(f"Accuracy: {metadata['accuracy']}, Direct Accuracy: {metadata['direct_accuracy']}")
        print(f"Prediction is: {metadata['prediction']} \nProgram ({'valid' if metadata['program_valid'] else 'invalid'}):\n{metadata['program']}")
        if not metadata['program_valid']:
            print(f"LLM Output: {metadata['llm_output']}")

    def eval(self, question, units, preds, answer, context=None, num_tokens=None, verbose=True):
        metadata = {
            "question": question,
            "units": units,
            "answer": answer,
            "context": "- " + "\n- ".join(context) if context is not None else "",
            "llm_output": preds,
            'prediction': 0.0,
            "direct_prediction": 0.0,
            'program': '',
            'program_valid': False,
        }
        accuracy = 0.0
        direct_accuracy = 0.0
        direct_out = None
        try:
            direct_out, program, context = self.split_context_program(preds.split("="))
            compiled_answer = compile_fp(context, program)
            compiled_out, compiled_units = convert_units(compiled_answer['P'])

            accuracy = accuracy_metric(answer, compiled_out)
            direct_accuracy = accuracy_metric(answer, direct_out)

            metadata = {
                **metadata,
                "prediction": compiled_out,
                "program": context + "=" + program,
                "program_valid": True,
            }

        except:
            if direct_out is not None:
                direct_accuracy = accuracy_metric(answer, direct_out)

            verbose = True

        metadata = {
                **metadata,
                "direct_prediction": direct_out,
                "accuracy": accuracy,
                "direct_accuracy": direct_accuracy,
            }

        if verbose:
            self.print_metadata(metadata)

        return accuracy, int(metadata["program_valid"]), metadata


    def parse_output(self, raw_output):
        return {
            "accuracy": raw_output[0],
            "parsable": raw_output[1],
            "metadata": raw_output[2],
        }

    def ask(self, question, units, answer, context=None, distractor_context=None, verbose=True):

        messages = [{"role": "system", "content": SYSTEM_PROMPT_STRING}]
        for few_shot_prompt in self.few_shot_prompts:
            user_question = few_shot_prompt.split("User:")[1].split("Assistant:")[0].strip('\n')
            messages.append({"role": "user", "content": user_question})
            assistant_response = "\n" + few_shot_prompt.split("Assistant:")[1].split("User:")[0].strip('\n')
            messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": QUERY_PROMPT.format(question=question, units=units, context="")})

        if verbose:
            print(f"\n{'-' * 110}\n{'-' * 45} NO CONTEXT MESSAGE {'-' * 45}\n{'-' * 110}\n\n")
        time.sleep(15)

        preds = self.chat(messages)

        no_ctxt_raw_output = self.eval(question, units, preds, answer, verbose=verbose)
        reg_ctxt_raw_output, dstr_ctxt_raw_output = (0, 0, {}), (0, 0, {})

        if context is not None:
            time.sleep(15)
            if verbose:
                print(f"\n{'-' * 115}\n{'-' * 45} PERFECT CONTEXT MESSAGE {'-' * 45}\n{'-' * 115}\n\n")
            messages[-1]["content"] = QUERY_PROMPT.format(question=question, units=units, context="- " + "\n- ".join(context))

            preds = self.chat(messages)
            reg_ctxt_raw_output = self.eval(question, units, preds, answer, context=context, verbose=verbose)

        if distractor_context is not None:
            time.sleep(15)
            if verbose:
                print(f"\n{'-' * 118}\n{'-' * 45} DISTRACTOR CONTEXT MESSAGE {'-' * 45}\n{'-' * 118}\n\n")
            messages[-1]["content"] = QUERY_PROMPT.format(question=question, units=units, context="- " + "\n- ".join(distractor_context))
            preds = self.chat(messages)
            dstr_ctxt_raw_output = self.eval(question, units, preds, answer, context=distractor_context, verbose=verbose)

        return (self.parse_output(raw_output) for raw_output in (no_ctxt_raw_output, reg_ctxt_raw_output, dstr_ctxt_raw_output))


    def run(self, dataset, N=None, output_store_path=None, verbose=False):
        N = N if N is not None else len(dataset)

        output_store = []
        noctxt_accuracy = 0.0
        regctxt_accuracy = 0.0
        dstrctxt_accuracy = 0.0
        iterator = tqdm(range(N), desc="Running Experiment 0")
        for i in iterator:
            entry = dataset[i]

            question = entry["question"]
            units = entry["units"]
            answer = entry["answer"]
            context = entry["context"].split('=')[1:]
            distractor_context = entry["distractor_context"].split('=')[1:]

            if verbose:
                print("About to go into ask")
            noctxt, regctxt, dstrctxt = self.ask(question, units, answer, context, distractor_context, verbose=verbose)

            output_store.append({
                "no-context": noctxt,
                "regular-context": regctxt,
                "distractor-context": dstrctxt,
            })

            if output_store_path is not None:
                with open(output_store_path, 'w') as f:
                    json.dump(output_store, f)

            noctxt_accuracy += noctxt["parsable"]
            regctxt_accuracy += regctxt["parsable"]
            dstrctxt_accuracy += dstrctxt["parsable"]

            iterator.set_postfix({"NoCtxt Valid": noctxt_accuracy/(i+1), "RegCtxt Valid": regctxt_accuracy/(i+1), "DstrCtxt Valid": dstrctxt_accuracy/(i+1)})


        accuracy = {
            "no-context": 0,
            "regular-context": 0,
            "distractor-context": 0,
        }
        parsable = {
            "no-context": 0,
            "regular-context": 0,
            "distractor-context": 0,
        }

        for output in output_store:
            for key in output:
                accuracy[key] += output[key]["accuracy"]
                parsable[key] += output[key]["parsable"]

        return {k: v/N for k, v in accuracy.items()}, {k: v/N for k, v in parsable.items()}, output_store



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    parser.add_argument('--model', type=str, default="gpt-4", help="OpenAI model to use")
    parser.add_argument('--temp', type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument('--fewshot_dataset', type=str, default=FEWSHOT_DATASET_PATH, help="Path to the few-shot dataset")
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help="Path to the dataset")
    parser.add_argument('--output_store', type=str, default=OUTPUT_STORE_DIR, help="Path to the output store")
    parser.add_argument('-k', type=int, default=5, help="Number of examples in the few-shot prompt")
    parser.add_argument('-N', type=int, default=-1, help="Number of test points")
    parser.add_argument('--save_each', action="store_true", default=True, help="Path to the output store")
    parser.add_argument('--verbose', action='store_true', help='Should the model be verbose?')

    args = parser.parse_args()
    np.random.seed(args.seed)

    with open(args.fewshot_dataset, 'r') as f:
        fewshot_dataset = json.load(f)

    few_shot_prompts = []
    for i in range(args.k):
        entry = fewshot_dataset[i]
        question = entry["question"]
        units = entry["units"]
        answer = entry["answer"]
        context = entry["context"].split('=')[1:]
        if args.k == 5:
            if i == 3:
                context = []
            elif i == 4:
                context = entry["distractor_context"].split('=')[1:]
        # context = []
        context = "- " + "\n- ".join(context) if len(context) > 0 else ""

        few_shot_prompts.append(f"User:{QUERY_PROMPT.format(question=question, units=units, context=context)}\nAssistant:{ASSISTANT_RESP_PROMPT.format(answer=entry['answer'], context=entry['context'], program=entry['program'])}")

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    predictor = SamplePredictor(few_shot_prompts=few_shot_prompts, model_name=args.model, temperature=args.temp)
    output_store_path = os.path.join(args.output_store, f"output_store_k={args.k}.json")
    acc, pars, output_store = predictor.run(
                                        dataset,
                                        N=args.N if args.N > 0 else None,
                                        output_store_path=output_store_path,
                                        verbose=args.verbose
                                    )

    print("Average Parsable Percentage: ", pars)
    print("Average Accuracy: ", acc)

    with open(output_store_path, 'w') as f:
        json.dump(output_store, f)
