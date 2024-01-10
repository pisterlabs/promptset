import jsonlines
import re
import random

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from tqdm.auto import tqdm
from runners.utils import chain_run_wrapper
from prompts.date_understanding import EXAMPLES_DATE_UNDERSTANDING_RATIONALE
from utils.utils import format_multiple_choice
from utils.constants import CHARACTERS
from datasets import load_dataset


TEMPLATE_RATIONALE = """
{examples}
Q: {question}
Options:
{formatted_choices}
Answer: Let's think step by step.
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
    # example: python runners/data_collector/date_understanding/get_rationale.py -o data/raw/date_understanding_codex_rationales.jsonl
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


    llm = OpenAI(model_name=args.model_name, stop=['[END]'], max_tokens=1200)
    prompte_rationale = PromptTemplate(
        input_variables=["examples", "question", "formatted_choices"],
        template=TEMPLATE_RATIONALE
    )
    chain_get_rationale = LLMChain(llm=llm, prompt=prompte_rationale, output_key="rationale", verbose=True)

    dset_name = "date_understanding"
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
            question = '\n'.join(data['inputs'].split('\n')[:-1])
            # date_understanding correct answer is always A. here we randomize it first
            choices = data['multiple_choice_targets']
            random.shuffle(choices)
            correct_choice = CHARACTERS[choices.index(data['targets'][0])]
            formatted_choices = format_multiple_choice(choices)

            new_split = split
            prompt_data = {
                "examples": EXAMPLES_DATE_UNDERSTANDING_RATIONALE,
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

            # date_understanding have very few steps, so we move from unused train samples to validation/test
            num_steps = len(out['rationale'].split('\n')) - 1
            if num_steps > 2:
                out['meta_data']['new_split'] = 'validation'

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