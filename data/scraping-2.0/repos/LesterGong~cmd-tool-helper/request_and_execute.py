import os
import subprocess
from openai import OpenAI

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

client = OpenAI(
    api_key="your api_key"
)


def chatgpt_request(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except:
        return "ChatGPT ERROR"


def construct_and_request(requirements):
    replacement_values = {
        '!<INPUT 0>!': requirements
    }
    with open('prompt/prompt_template_v2.txt', 'r') as file:
        content = file.read()
        parts = content.split('<commentblockmarker>###</commentblockmarker>')
        prompt = parts[1]
        for placeholder, values in replacement_values.items():
            prompt = prompt.replace(placeholder, values)

    response = chatgpt_request(prompt)
    return response


def execute(cmd_line):
    completed_process = subprocess.run(cmd_line, text=True, capture_output=True, shell=True)
    return completed_process.stdout


# requirement = 'List all files in previous folder'
# response = construct_and_request(requirement)
# print(response)
# # print(type(response))
