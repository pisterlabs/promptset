import os
import json
import pathlib

import openai
from pathlib import Path

ENGINE = "gpt-3.5-turbo"
TEMPERATURE = 0

user1_msg = """Given the sentence 'Sarah was a much better surgeon than Maria so ___ always got the easier cases.', write a Python code that determines which of the following options correctly fills in the blank:
1) Sarah
2) Maria
"""
user2_msg = """Given the sentence 'Terry tried to bake the eggplant in the toaster oven but the ___ was too big', write a Python code that determines which of the following options correctly fills in the blank:
1) eggplant
2) toaster
"""

user_temp = """Given the sentence '{sentence}', write a Python code that determines which of the following options correctly fills in the blank:
1) {option1}
2) {option2}
"""

current_dir = pathlib.Path(__file__).parent.resolve()
dataset_path = Path(os.path.join(current_dir, "dataset/dev.jsonl"))
prediction_path = Path(os.path.join(current_dir, "results/simple-predictions.json"))


def process_dataset():
    sample_path = Path(os.path.join(current_dir, "results/sample1.py"))
    code_sample_file = open(sample_path, 'r')
    sample1_code = code_sample_file.read()

    sample_path = Path(os.path.join(current_dir, "results/sample2.py"))
    code_sample_file = open(sample_path, 'r')
    sample2_code = code_sample_file.read()

    dataset_items = list(open(dataset_path, 'r'))
    for str_item in dataset_items[10:110]:
        data = json.loads(str_item)
        qID = data['qID']
        sentence = data['sentence']
        option1 = data['option1']
        option2 = data['option2']
        code_representation_path = Path(
            os.path.join(current_dir, "results/code/{engine}/{hash}.py".format(engine=ENGINE, hash=qID)))

        if not code_representation_path.is_file():
            response = openai.ChatCompletion.create(
                model=ENGINE,
                messages=[

                    {"role": "user", "content": user1_msg},
                    {"role": "assistant", "content": sample1_code},

                    {"role": "user", "content": user2_msg},
                    {"role": "assistant", "content": sample2_code},
                    {"role": "user", "content": user_temp.format(sentence=sentence, option1=option1, option2=option2)},

                ],
                temperature=TEMPERATURE,
            )
            output = response['choices'][0]['message']['content']

            with open(code_representation_path, 'w') as f:
                f.write(output)
            print(qID, "finished")


process_dataset()
