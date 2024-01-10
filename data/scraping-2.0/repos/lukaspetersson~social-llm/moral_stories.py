from datasets import load_dataset
import openai
from dotenv import dotenv_values
import json

env_var = dotenv_values(".env")
openai_token = env_var["OPENAI_TOKEN"]
openai.api_key = openai_token


##

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

data_path = "./data/moral_stories_full.jsonl"
data = read_jsonl(data_path)

## Naive approach
setting = "You will be presented with a moral dilemma. Your task is to suggest actions to take in the given situation. Present each action in at most 2 sentences. Each action you present should be different. Important: each suggested action should be based on the same assumptions of the situation.\nThe situation is as follows:\n"
scenario = data[0]["situation"] 
instruction = "Present 2 different actions. One that is the most moral, and one that most people would do."

messages = [
    {"role": "system", "content": setting+scenario},
    {"role": "user", "content": instruction},
]

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)
print(resp.choices[0].message.content)

## AI Feedback (Anthropic inspired)
moral_action = "Present an action to the situation that you think is most moral"
messages = [
    {"role": "system", "content": setting+scenario},
    {"role": "user", "content": moral_action},
]
resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)
messages.append(resp.choices[0].message)

critique_request = " Identify a perspective from which the assistant’s last response is not the most moral."
messages.append({"role": "user", "content": critique_request})

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)
messages.append(resp.choices[0].message)

revision_request = "Please rewrite the assistant response to make it more moral from this new perspective."
messages.append({"role": "user", "content": revision_request})

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)

for msg in messages:
    print(msg["role"])
    print(msg["content"])
    print()
print("assistant")
print(resp.choices[0].message.content)


## Predefined perspecives
perspectives = {
    "Ethics": [
        "Utilitarianism",
        "Deontology",
        "Virtue Ethics",
        "Contractarianism",
        "Care Ethics",
        "Egoism"
    ],
    "Values": [
        "Honesty",
        "Compassion",
        "Loyalty",
        "Responsibility",
        "Fairness",
        "Respect",
        "Integrity"
    ],
    "Demographics": [
        "Age",
        "Gender",
        "Socioeconomic Status",
        "Education",
        "Ethnicity",
        "Religion",
        "Geographic Location"
    ]
}

## Ethics
ethics_instr = "For each of the following philosophies of ethics ("+", ".join(perspectives["Ethics"])+"), answer what the best action to the moral dilemma is, given that you follow that ethics. "
messages = [
    {"role": "system", "content": setting+scenario},
    {"role": "user", "content": ethics_instr},
]

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)
print(resp.choices[0].message.content)

## Values
values_instr = "For each of the following values ("+", ".join(perspectives["Values"])+"), answer what the best action to the moral dilemma is, given that you follow that value."
messages = [
    {"role": "system", "content": setting+scenario},
    {"role": "user", "content": values_instr},
]

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)
print(resp.choices[0].message.content)

## Demographics
demographics_instr = "For each of the following demographics types ("+", ".join(perspectives["Demographics"])+"), make up two different personas and answer what the best action to the moral dilemma is for each. Give only one action per persona."
messages = [
    {"role": "system", "content": setting+scenario},
    {"role": "user", "content": demographics_instr},
]

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
)
print(resp.choices[0].message.content)

## Filter
def filter_actions(messages):
    filter_msg = "The goal of generating these actions was to highlight the fact that there are often many moral actions to a given dilemma. To present this as clearly as possible, it is important that the actions we present are reasonable and are all different from each other. Modify the list of actions previously proposed by the assistant by excluding actions that don't fit with this goal. Keep at most  3 actions."
    messages.append({"role": "user", "content": filter_msg})
    return openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
    )

messages.append(resp.choices[0].message)
filtered_resp = filter_actions(messages)
print(filtered_resp.choices[0].message.content)
