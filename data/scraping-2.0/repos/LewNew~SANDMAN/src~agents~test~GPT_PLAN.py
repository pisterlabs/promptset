import datetime
import os
import random

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
import json
from ElementsDict import *


def chat(system, user_assistant, max_tokens):
    assert isinstance(system, str), "`system` should be a string"
    assert isinstance(user_assistant, list), "`user_assistant` should be a list"

    system_msg = {"role": "system", "content": system}

    user_assistant_msgs = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for i, msg in enumerate(user_assistant)
    ]

    messages = [system_msg] + user_assistant_msgs

    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=messages)

    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"The status code was {status_code}."

    return response["choices"][0]["message"]["content"]


max_tokens_limit = 25
output_file_path = '../models/chat_outputs.json'

# Check if the file exists and is not empty
if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
    with open(output_file_path, 'r') as output_file:
        output_data = json.load(output_file)
else:
    output_data = []

# Need to find the highest prompt number in output JSON so we don't duplicate it
highest_prompt_number = max([0] + [item['Prompt'] for item in output_data])

# Update for number of iterations (prompts)
num_prompts = 5

for i in range(highest_prompt_number + 1, highest_prompt_number + 1 + num_prompts):
    role = random.choice(roles)
    organization = random.choice(organizations)
    topic = random.choice(topics)
    selected_style = random.choice(style)

    random_prompt = f'You are a {role} in a {organization} that is writing about {topic} in the style of {style}'

    initiation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if selected_style in style:
        try:
            response = chat(random_prompt, [''], max_tokens_limit)
            output_data.append({
                "Prompt": i,
                "Initiated": initiation_time,
                "Role": role,
                "Organization": organization,
                "Topic": topic,
                "Style": selected_style,
                "Response": response
            })
        except ValueError as e:
            output_data.append({
                "Prompt": i,
                "Initiated": initiation_time,
                "Error": str(e)
            })
    else:
        output_data.append({
            "Prompt": i,
            "Initiated": initiation_time,
            "Error": f"Invalid style: {style}"
        })

# Save the updated data to the JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(output_data, output_file, indent=4)

print(f"JSON saved to {output_file_path} with {num_prompts} prompts")

for i, prompt_data in enumerate(output_data[-num_prompts:]):
    print(
        f"Prompt {i + 1}: {prompt_data['Role']} - {prompt_data['Organization']} - {prompt_data['Topic']} - {prompt_data['Style']}")
