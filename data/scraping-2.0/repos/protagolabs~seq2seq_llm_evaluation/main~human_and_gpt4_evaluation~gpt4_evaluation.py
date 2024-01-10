"""
This script is used to reproduce the GPT-4 model-to-model evaluation.
Note that the data input files have not been included in this repository and need to be generated
using this codebase and the datasets appropriately downloaded as explained in the README.md
"""
import os
import time
import json
import argparse
from datetime import datetime
import openai
import pandas as pd
import pytz
import ndjson


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", help="The task to use. Can be 'summarization', 'simplification' or 'GEC'.",
        choices=["summarization", "simplification", "GEC"], required=True
    )
    args = parser.parse_args()
    TASK = args.task  # 'simplification', 'summarization' or 'GEC'
    with open(
            f'main/human_and_gpt4_evaluation/instructions_to_human_reviewers_and_gpt4/'
            f'gpt4_eval_{TASK}_instructions.txt',
            'r'
    ) as f:
        initial_prompt = f.read()
    df = pd.read_csv(f'data/outputs/{TASK}_evaluation/data_files/text_{TASK}_potato.csv')
    # Note: this is the output with the predictions from all the models. It is not included in this
    # repository but it can be reproduced using this codebase
    data_to_evaluate = df.to_dict()
    formatted_data = [{key: data_to_evaluate[key][i] for key in data_to_evaluate.keys()} for i in range(len(df))]
    openai.api_key = os.getenv('OPENAI_API_KEY')  # you should place your API key here
    output = []

    full_log = []

    for input_sample in formatted_data:
        inference_not_done = True
        while inference_not_done:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": initial_prompt
                        },
                        {
                            "role": "assistant",
                            "content": "This is clear, and I have no questions. I am ready to start."
                        },
                        {
                            "role": "user",
                            "content": str(input_sample)
                        }
                    ]
                )

                output.append(json.loads(completion['choices'][0]['message']['content']))
                full_log.append(completion.to_dict())
                with open(f'data/outputs/{TASK}_evaluation/gpt4_as_annotator/full_log.jsonl', 'w') as f:
                    ndjson.dump(full_log, f)
                with open(f'data/outputs/{TASK}_evaluation/gpt4_as_annotator/output.jsonl', 'w') as f:
                    ndjson.dump(output, f)
                inference_not_done = False
            except Exception as e:  # pragmatic catch all exception is not ideal, but works well for now as we don't
                # know which error OpenAI API will throw (it is still unstable and can throw many different errors)
                # We retry after 10 minutes as often the OpenAI server will start working again
                print(f"Waiting 10 minutes, current time: {datetime.now(pytz.timezone('Europe/London')).isoformat()}")
                print(f"Error was: {e}")
                time.sleep(600)
