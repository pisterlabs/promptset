import os
import json
import random
import numpy as np
from llm_utils import OpenAI

prompt = """
You will receive a list of questions or instructions. Your objective is to determine if it satisfies the following requirements:
1. It can be completed without relying on external, up-to-date factual knowledge.
2. It does not require any additional input.
3. It is not related to coding or solving complex math problems.
4. It is not a question with personal feeling or opinion.
Please only respond with a clear YES if all the requirements are met for a task, or respond with a NO if any of the requirements are not satisfied. 
If a question is not valid, specify which requirement it violates. Avoid providing any additional information in your response.

Example 1:
Q: What is the weather like right now?
A: NO
Violation: 1

Example 2:
Q: Create a simple calculator using javascript.
A: NO
Violation: 3

Example 3:
Q: How to make a cake?
A: YES
Violation: None

Example 4: 
Q: Write a short summary of this article.
A: NO
Violation: 2

Example 5:
Q: Generate a list of all the possible 4-letter words that start with \"d\".
A: YES
Violation: None

Example 6:
Q: How do you think of trump?
A: NO
Violation: 4

Example 7:
Q: {task}
A:
"""

"""
First filter: remove instruction:
1. require input
2. involve coding, translation
"""
uniq_instructions = []
filtered_instruction_data_path = f"./data/qa/self-instruct/rule_filtered_instances_82K.jsonl"
if os.path.exists(filtered_instruction_data_path):
   with open(filtered_instruction_data_path, "r") as file:
    for line in file:
        instruction = json.loads(line)
        # Process the JSON object for each line
        uniq_instructions.append(instruction['instruction'])
else:
    instructions = []
    uniq_instructions = []
    instruction_data_path = f"./data/qa/self-instruct/all_instances_82K.jsonl"
    with open(instruction_data_path, 'r') as f:
        for line in f:
            instruction = json.loads(line)
            # filter out the instruction about coding, translation
            if not instruction['input'] and \
                instruction['instruction'] not in uniq_instructions and \
                "function" not in instruction['instruction'] and \
                "algorithm" not in instruction['instruction'] and \
                "web" not in instruction['instruction'] and \
                "you" not in instruction['instruction'] and \
                "import" not in instruction['output'] and \
                "def " not in instruction['output'] and \
                "Translate" not in instruction['instruction']: 
                    instructions.append(instruction)
                    uniq_instructions.append(instruction['instruction'])

    with open(filtered_instruction_data_path, 'w', encoding='utf-8') as f:
        for json_obj in instructions:
            json_line = json.dumps(json_obj, ensure_ascii=False)
            f.write(json_line + '\n')

"""
Second filter with gpt3
"""
gpt3_filtered_instruction_data_path = f"./data/qa/self-instruct/gpt3_filtered_instances_82K.jsonl"
if os.path.exists(gpt3_filtered_instruction_data_path):
    gpt3_filtered_instructions = []
    with open(instruction_data_path, 'r') as f:
        for line in f:
            instruction = json.loads(line)
            gpt3_filtered_instructions.append(instruction)
else:
    gpt3_filtered_instructions = []

    gpt3 = OpenAI(model="gpt-3.5-turbo-1106")
    valid_num = len(uniq_instructions)

    random.shuffle(uniq_instructions)
    for instruction in uniq_instructions:    
        input = prompt.format(task=instruction)
        output = gpt3.generate(prompt=input,
                                temperature=0.7, 
                                top_p=1.0, 
                                max_tokens=20, 
                                n=1, 
                                frequency_penalty=0, 
                                presence_penalty=0, 
                                stop=["Example"])[0]
        try:
            output = output.split("Violation:")
            valid = output[0].strip().lower()
            violation = output[1].strip()

            # whether valid task
            if valid == 'yes':
                valid = 1
            elif valid == 'no':
                valid = 0
            else:
                raise TypeError
            
            # violation type
            if violation == '1':
                violation = 1
            elif violation == '2':
                violation = 2
            elif violation == '3':
                violation = 3
            elif violation == '4':
                violation = 4
            elif violation == 'None':
                violation = 0
            else:
                raise TypeError
            
            gpt3_filtered_instructions.append(
                {
                    "instruction": instruction,
                    "valid": valid,
                    "type": violation
                }
            )

            print(valid, violation)
            if valid: valid_num += 1
        except:
            continue
        
        if valid_num == 1000:
            break
        
    with open(filtered_instruction_data_path, 'w', encoding='utf-8') as f:
        for json_obj in gpt3_filtered_instructions:
            json_line = json.dumps(json_obj, ensure_ascii=False)
            f.write(json_line + '\n')
