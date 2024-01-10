import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import openai
import tiktoken

from settings import Settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "text-davinci-003"
CODE_BASE_PATH = '../code-base/{engine}/{ques_id}.py'
CODE_UPDATED_PATH = '../code-updated/{engine}/{ques_id}.py'
SLEEP_TIME = 100
MAX_TOKENS = 4097
TEMPERATURE = 0.3

openai.api_key = OPENAI_API_KEY


def generate_para_steps_prompt(steps):
    para_steps_prompt = "I have a procedural event with the following steps. " \
                        "Create a python code for it which has different classes for" \
                        " the participants and their relationship with each other.\n"
    for step in steps:
        para_steps_prompt += f'- {step}\n'
    return para_steps_prompt


def generate_stem_prompt(stem):
    stem_prompt = f"Update the base code you created with following question : {stem}.\n" \
                  "The code should be executable so I run the updated code to get the answer of the question.\n" \
                  "Valid options : ['more', 'less', 'no effect']"
    return stem_prompt


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_code_representation_from_gpt(prompt):
    num_tokens = num_tokens_from_string(prompt, MODEL_NAME)

    response = openai.Completion.create(
        engine=MODEL_NAME,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS - num_tokens,
    )
    return response.choices[0].text.strip()


def process_dataset():
    with open(Settings.wiqa_train_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        data = json.loads(json_str)
        ques_id = data['metadata']['ques_id']
        code_base_representation_path = Path(CODE_BASE_PATH.format(engine=MODEL_NAME, ques_id=ques_id))
        code_updated_representation_path = Path(CODE_UPDATED_PATH.format(engine=MODEL_NAME, ques_id=ques_id))

        if not code_base_representation_path.is_file():
            para_steps = data['question']['para_steps']
            para_steps_prompt = generate_para_steps_prompt(para_steps)

            code_representation = get_code_representation_from_gpt(para_steps_prompt)
            with open(code_base_representation_path, 'w') as f:
                f.write(code_representation)
            print(ques_id, "finished")

            # Avoid hitting rate limit
            time.sleep(SLEEP_TIME)

            stem = data['question']['stem']
            stem_prompt = generate_stem_prompt(stem)

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": para_steps_prompt},
                    {"role": "assistant", "content": code_representation},
                    {"role": "user", "content": stem_prompt},
                ],
                temperature=0,
            )

            code_updated_representation = response['choices'][0]['message']['content']

            with open(code_updated_representation_path, 'w') as f:
                f.write(code_updated_representation)
            print(ques_id, "finished")


process_dataset()
