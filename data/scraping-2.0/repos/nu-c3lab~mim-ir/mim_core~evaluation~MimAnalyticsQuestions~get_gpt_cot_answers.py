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

import argparse, json, re
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
            while not question_answered:
                try:
                    prompt = "Example" \
                    + "\nQuestion: Which country contains the river that is longest among those in the continent having the tallest mountain?" \
                    + "\nGive the answer as a single noun phrase. Let's think step by step." \
                    + "\n\nFirst, we need to identify the continent with the tallest mountain. The tallest mountain in the world is Mount Everest, which is located in the continent of Asia." \
                    + "\n\nNext, we need to determine which country contains the longest river in Asia. The longest river in Asia is the Yangtze River, which is located in China." \
                    + "\n\nAnswer: China" \
                    + "\n\nUsing the form of the example above, answer the following question." \
                    + f"\nQuestion: {sample['question']}" \
                    + "\nGive the answer as a single noun phrase. Let's think step by step."

                    result = model.generate(prompt)
                    match = re.search(r'Answer: (.*)$', result)
                    answer = match.group(1) if match else 'no'

                    output['answers'][sample['id']] = answer
                    print(sample['question'])
                    print(answer)
                    print('-------------------------------')
                    question_answered = True
                except RateLimitError as e:
                    print('Hit rate limit. Retrying')
                except:
                    output['answers'][sample['id']] = 'error'
                    question_answered = True

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
