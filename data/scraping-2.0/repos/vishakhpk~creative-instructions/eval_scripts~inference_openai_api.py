import os
import openai
import sys
import json 
import copy
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt_file = sys.argv[1]
instruction_file = sys.argv[2]
output_file = sys.argv[3]

with open(prompt_file, 'r') as f:
    prompt = f.read()

ip = []
with open(instruction_file, 'r') as f:
    for line in f:
        ip.append(json.loads(line.strip()))

print(type(prompt))
print(prompt)

op = []
for item in ip:
    instruction = item['translation']['en1']

    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt + instruction,
        temperature=0.7,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["."]
        )

    op_item = copy.deepcopy(item)
    gen = response['choices'][0]['text']
    gen = gen.strip().split(":")[-1].strip()
    op_item['generation'] = gen #response['choices'][0]['text']
    op_item['response'] = response
    op_item['instruction'] = instruction
    # print(response)
    op.append(op_item)
    time.sleep(1)

with open(output_file, "w") as f:
    for item in op:
        f.write(json.dumps(item)+'\n')
