import datetime
import os
import random
from openai import OpenAI
from ..ElementsDict import *

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Trying to avoid generating too many API keys because I'm not tracking them.
# Set OPEN_API_KEY as an environment variable in CMD prompt with setx OPEN_API_KEY "<KEY>" or do it manually
# Get your own key ya slugs


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


# Define the maximum token limit you want to set
max_tokens_limit = 25  # Set your desired token limit here

# Adjust i in range(x) value for n times the prompt function iterates

output_file_path = '../models/chat_outputs.txt'
with open(output_file_path, 'w') as output_file:
    for i in range(2):
        role = random.choice(roles)
        organization = random.choice(organizations)
        topic = random.choice(topics)

        # We can plug in literally anything here to give the prompt context
        # Update ElementsDict lists
        random_prompt = f'You are a {{}} in a {{}} that is writing about {{}} in the style of {{}}'.format(role,
                                                                                                           organization,
                                                                                                           topic, style)

        initiation_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            response = chat(random_prompt, [''], max_tokens_limit)
            output_file.write(
                f'Prompt {{i+1}}: {random_prompt}\nInitiated: {initiation_time}\nResponse: {response}\n\n')
        except ValueError as e:
            output_file.write(f'Error in Prompt {{i+1}}: {e}\n')
