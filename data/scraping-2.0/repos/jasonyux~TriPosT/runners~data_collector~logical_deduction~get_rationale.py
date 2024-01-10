import jsonlines
import re
import os
import random

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from tqdm.auto import tqdm
from runners.utils import chain_run_wrapper
from prompts.logical_deduction import EXAMPLES_LOGICAL_DEDUCTION_RATIONALE
from utils.utils import format_multiple_choice
from utils.constants import CHARACTERS
from datasets import load_dataset


TEMPLATE_RATIONALE = """
{examples}
Q: {question}
Options:
{formatted_choices}
Answer: Let's think step by step. Let "??" represents 0 or more objects, and "?" represents exactly 1 object.
""".strip()


def get_total(dataset, args):
    total = 0
    for split, dset in dataset.items():
        if split == 'default':
            continue
        for _ in dset:
            total += 1
    return total


if __name__ == "__main__":
    # example: python runners/data_collector/logical_deduction/get_rationale.py -o data/raw/logical_deduction_codex_rationales.jsonl
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-Companion")
    parser.add_argument(
        "-o",
        type=str,
        help="jsonl output file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="code-davinci-002",
        help="The name of the model",
    )
    args = parser.parse_args()


    llm = OpenAI(model_name=args.model_name, stop=['[END]'], max_tokens=2048)
    prompte_rationale = PromptTemplate(
        input_variables=["examples", "question", "formatted_choices"],
        template=TEMPLATE_RATIONALE
    )
    chain_get_rationale = LLMChain(llm=llm, prompt=prompte_rationale, output_key="rationale", verbose=True)

    dset_name = "logical_deduction"
    dataset = load_dataset("bigbench", dset_name)
    results = []
    _acc = {'total': 0, 'correct': 0}
    total_ = get_total(dataset, args)
    pbar = tqdm(total=total_, desc="Generating Rationale")
    random.seed(42)
    for split, dset in dataset.items():
        if split == 'default':
            continue

        for data in dset:
            _acc['total'] += 1
            question = data['inputs'].replace('\n\n', '\n').strip()
            choices = data['multiple_choice_targets']
            correct_choice = CHARACTERS[data['multiple_choice_scores'].index(1)]
            formatted_choices = format_multiple_choice(choices)

            new_split = split
            prompt_data = {
                "examples": EXAMPLES_LOGICAL_DEDUCTION_RATIONALE,
                "question": question,
                "formatted_choices": formatted_choices,
                "correct_choice": correct_choice,
                'meta_data': {
                    'split': split,
                    'new_split': new_split,
                    'idx': data['idx'],
                }
            }
            out = chain_run_wrapper(chain_get_rationale, prompt_data)
            out['rationale'] = out['rationale'].strip()

            # check if the answer is correct
            final_answer = out['rationale'].split('\n')[-1]
            final_choice = re.search(r"answer is \(([a-zA-Z])\).*", final_answer)
            
            if final_choice is None or len(final_choice.groups()) == 0:
                pbar.update(1)
                continue
            if final_choice.groups()[0].lower() != correct_choice.lower():
                pbar.update(1)
                continue
            _acc['correct'] += 1
            
            # save the result
            results.append(out)
            jsonlines.open(args.o, 'w').write_all(results)

            pbar.update(1)
            pbar.set_postfix({'acc': f"{_acc['correct'] / _acc['total']:.4f}"})
    pbar.close()