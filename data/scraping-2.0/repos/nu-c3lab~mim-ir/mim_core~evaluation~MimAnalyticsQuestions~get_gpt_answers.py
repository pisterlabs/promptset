'''
This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim.
If not, see <https://www.gnu.org/licenses/>.
'''
"""
Mim Analytics Dataset Evaluation

Authors: C3 Lab

Example Usage:

    python get_answers.py data/analytic_data.json test_1

Or if you want to test only a subset of the operators:

    python get_answers.py data/analytic_data.json test_1 --ops addition discard subtraction

"""

import json, time
import argparse
from tqdm import tqdm
from typing import List
from mim_core.components.Search.GPT35Interface import GPT35Interface
from openai.error import RateLimitError

def main(question_file: str,
         output_postfix: str,
         operators: List[str]):

    # Init the GPT Interface
    model = GPT35Interface()

    # Init the output dictionary
    output = {
        'answers': {}
    }

    plans = {}

    # Read in the json file of the HotpotQA questions
    with open(question_file) as f:
        # Read in the question file
        data = json.load(f)

        final_data = [ex for ex in data if ex["operator"] in operators]

        for sample in tqdm(final_data):
            question_answered = False
            backoff = 1
            while not question_answered:
                time.sleep(backoff)
                try:
                    prompt = 'Answer the following question with a single noun phrase.'
                    prompt += f'\n\nQuestion: {sample["question"]}'
                    prompt += '\nAnswer: '
                    answer = model.generate(prompt)
                    output['answers'][sample['id']] = answer
                    print(sample['question'])
                    print(answer)
                    print('-------------------------------')
                    question_answered = True
                    backoff = 1
                except RateLimitError as e:
                    question_answered = False
                    backoff *= 2
                    print(f'Hit rate limit reached. Retrying after {backoff} secs.')
                except:
                    output['answers'][sample['id']] = 'error'
                    # Count other errors as an answered question
                    question_answered = True
                    backoff = 1

    # Write out the answers to a file for evaluation by the official HotpotQA script
    with open("results/answers_{}.json".format(output_postfix), "w") as outfile:
        outfile.write(json.dumps(output, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question_file", type=str)
    parser.add_argument("output_postfix", type=str)
    parser.add_argument("--ops", nargs="*", default=["addition", "boolean_equality", "boolean_existence", "bridge", "comparison", "discard", "intersection", "subtraction"])
    args = parser.parse_args()

    main(args.question_file, args.output_postfix, args.ops)
