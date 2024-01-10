import json
import os

import openai
from dotenv import load_dotenv

from settings import Settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
# {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
# {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
# ...

with open(Settings.wiqa_train_path, 'r') as json_file:
    train_data = [json.loads(line) for line in json_file]

with open('output.jsonl', 'w') as outfile:
    for data in train_data:
        ques_id = data['metadata']['ques_id']
        para_steps = " ".join([p.strip() for p in data["question"]["para_steps"] if len(p) > 0])
        stem = data['question']['stem'].strip()
        stem = stem.replace(".", "")
        answer_label = data['question']['answer_label'].strip()

        prompt_text = f"{para_steps}. " \
                      f"{stem} \n\n"
        json_item = {"prompt": prompt_text, "completion": " " + answer_label + "###"}

        json.dump(json_item, outfile)
        outfile.write('\n')
