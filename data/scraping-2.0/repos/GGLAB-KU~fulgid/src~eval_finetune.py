import json
import os
import re

import openai
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import f1_score

from settings import Settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
y_true = []
y_pred = []

with open(Settings.wiqa_test_path, 'r') as json_file:
    test_data = [json.loads(line) for line in json_file]

for data in tqdm(test_data):
    ques_id = data['metadata']['ques_id']
    para_steps = " ".join([p.strip() for p in data["question"]["para_steps"] if len(p) > 0])
    stem = data['question']['stem'].strip()
    stem = stem.replace(".", "")
    answer_label = data['question']['answer_label'].strip()

    prompt_text = f"{para_steps}. " \
                  f"{stem} \n\n"

    response = openai.Completion.create(
        engine="curie:ft-koc-university-2023-06-15-15-03-52",
        prompt=prompt_text,
    )
    predicted_answer = response.choices[0].text.strip()
    pattern = r'^(more|less|no_effect)'
    string = "more text"

    match = re.match(pattern, predicted_answer)
    if match:
        starting_word = match.group(1)
        y_pred.append(starting_word)
    else:
        y_pred.append(predicted_answer)
        print("String does not match the pattern.")

    y_true.append(answer_label)

# Calculate the F1 score
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'F1 score: {f1}')

print(y_pred)
