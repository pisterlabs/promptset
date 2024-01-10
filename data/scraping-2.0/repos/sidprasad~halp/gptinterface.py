import openai
from openai import OpenAI
import json


## Client Setup
 # Fix this somehow


with open('secrets.json') as f:
    secrets = json.load(f)

API_KEY = secrets['API_KEY']
client = OpenAI(api_key=API_KEY)


            
def split_prompt(text, split_length, role):
    if split_length <= 0:
        raise ValueError("Max length must be greater than 0.")

    num_parts = -(-len(text) // split_length)
    file_data = []

    for i in range(num_parts):
        start = i * split_length
        end = min((i + 1) * split_length, len(text))

        # if i == num_parts - 1:
        #     content = f'[START PART {i + 1}/{num_parts}]\n' + text[start:end] + f'\n[END PART {i + 1}/{num_parts}]'
        #     content += '\nALL PARTS SENT. Now you can continue processing the request.'
        # else:
        #     content = f'Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {i + 1}/{num_parts} received" and wait for the next part.\n[START PART {i + 1}/{num_parts}]\n' + text[start:end] + f'\n[END PART {i + 1}/{num_parts}]'
        #     content += f'\nRemember not answering yet. Just acknowledge you received this part with the message "Part {i + 1}/{num_parts} received" and wait for the next part.'

        # Either this or the stuff above?
        content = text[start:end] 

        file_data.append({
            'role': role,
            'content': content
        })

    return file_data


def ask_gpt_chunked(system_prompt, user_prompt):
    sys_msgs = split_prompt(system_prompt, 2500, role = 'system')
    user_messages = split_prompt(user_prompt, 2500, role = 'user')
    messages = sys_msgs + user_messages
    return ask_gpt(messages)
    


def ask_gpt(messages):

    # Need to modify the messages array to remember this stuff.
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages)
    return completion.choices[0].message.content


