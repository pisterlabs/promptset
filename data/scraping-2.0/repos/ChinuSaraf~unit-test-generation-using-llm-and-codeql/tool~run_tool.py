import json
import sys
import argparse
import openai
import datetime
import time
from datetime import date
import re
from prompt.tests import Tests
from helpers.utils import get_metadata, extract_class_from_class_qlf_name

# GET YOUR OPEN AI API KEY FROM : https://platform.openai.com/account/api-keys
# Save the key in config.json as { "OPENAI_API_KEY": "<KEY>" }


with open("config.json", "r") as file:
    config = json.loads(file.read())

openai.api_key = config["OPENAI_API_KEY"]


def chat(i, task, out, meta_data_arr):
    init = task.get_init_prompt()
    print(f">>> INIT: {init}")
    out.append("## Prompt")
    out.append("**" + init + "**")
    idx = 1
    instruction_str = ""
    repeat = False
    itr = 1
    task_prompt = [
        {
            "role": "system",
            "content": init
        }
    ]

    while True:
        # Skip the prompts which don't provide meta-data.
        if idx <= 4 and meta_data_arr[idx-1] == "":
            idx += 1
            continue

        # Beacuse of the Rate limit applied by OpenAI: 3 requests per minute
        if itr % 3 == 0:
            time.sleep(60)

        # Get prompt from the multi-prompt list
        prompt = task.generate_prompt(idx)

        # To print the method body inside a code block in the output
        if idx == 1:
            out.append("**" + prompt + "**")
            out.append('CODE:')
            out.append('```')
            out.append(task.get_code())
            out.append('```')
        else:
            out.append("## Prompt")
            out.append("**" + prompt + "**")

        # Add the supplementary information depending upon the current prompt
        if idx == 1:
            prompt += '\nCODE: \n' + task.get_code()
        elif idx == 2:
            prompt += '\nPARENT_CLASS: \n' + meta_data_arr[idx-1]
            out.append('\nPARENT_CLASS: \n' + meta_data_arr[idx-1])
            instruction_str += "PARENT_CLASS"
        elif idx == 3:
            prompt += '\nPARAMETERS: \n' + meta_data_arr[idx-1]
            out.append('\nPARAMETERS: \n' + meta_data_arr[idx-1])
            instruction_str += ", PARAMETERS"
        elif idx == 4:
            prompt += '\nLOCAL_VARIABLES: \n' + meta_data_arr[idx-1]
            out.append('\nLOCAL_VARIABLES: \n' + meta_data_arr[idx-1])
            instruction_str += ", LOCAL_VARIABLES"
        elif idx == 5:
            prompt = prompt.replace("<META-DATA>", instruction_str)
            if repeat:
                prompt += "\nGenerate the JUnit of the method completely!"

        if idx <= 4:
            prompt += f"\nIf you understand this, say only 'Yes'."
        task_prompt.append({
            "role": "user",
            "content": prompt
        })

        print(f">>> Prompt: {prompt}")

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=task_prompt
        )

        reply = chat.choices[0].message.content

        print(f"Reply: {reply}")

        # Log this response in markdown
        out.append("## Response")
        out.append("```")
        out.append(reply)
        out.append("```")

        if "no" in reply.lower()[:2]:
            print(f'Retrying as reply returned No')
            out.append("`**Note:** Retrying as ChatGPT output No`")
            out.append("---")
            continue

        if idx == 5:
            if not repeat and ("<JunitTest>" not in reply or "</JunitTest>" not in reply):
                repeat = True
                out.append(
                    "`**Note:** Retrying as ChatGPT did not output the entire code`")
                out.append("---")
                continue
            task.write_unit_test(
                str(i) + "-" + str(int(time.time())), reply)

        task_prompt.append({"role": "assistant", "content": reply})

        idx += 1
        itr += 1
        if not task.parse_response(idx, reply):
            task.store(str(i) + "-" + str(int(time.time())), out)
            break


def tests(i, data_json, out, meta_data_arr, mode_of_exec):
    out.append("# Task: Generate Tests")
    out.append(f"## Mode of Execution:")
    out.append(f"{mode_dict[mode_of_exec]}")
    out.append("---")
    test = Tests(i, data_json, mode_of_exec)
    chat(i, test, out, meta_data_arr, )


def read_data(data_path):
    # Load Dataset
    with open(data_path, encoding='utf8') as file:
        data = json.load(file)

    return data


# Script begins here

# Mode 0 : Without Meta-data, Mode 1: With only method's class meta-data, Mode 2: With only method_params,
# Mode 3: With only method_vars, Mode 4: With all Meta-data
# Default: Mode 4
mode_dict = {
    0: "Running the script without any Meta-data (Check `./output/generated-tests/wo-metadata` folder)",
    1: "Running the script with only the Method's Class Meta-data (Check `./output/generated-tests/class-metadata` folder)",
    2: "Running the script with only the Method's Parameter Meta-data (Check `./output/generated-tests/method-params` folder)",
    3: "Running the script with only the Method's Variable Meta-data (Check `./output/generated-tests/method-vars` folder)",
    4: "Running the script with all the Meta-data- Class, Method Parameters, Method Variables (Check `./output/generated-tests/all` folder)"
}

p = argparse.ArgumentParser()

p.add_argument('--mode', dest='mode',
               default=4, help='Describe the mode of execution', type=int)
p.add_argument('--n', dest='num_datapoints',
               default=1, help='Enter the number of data points to run', type=int)

args = p.parse_args()

# Get mode of execution
mode_of_exec = args.mode
# Get num of data points
num_datapoints = args.num_datapoints

data_path = "../data/methods.json"
data_json = read_data(data_path)
is_first = True

# If invalid mode selected, change it to default
if mode_of_exec >= 5 or mode_of_exec < 0:
    print(f'Selected Invalid Mode of Execution, so running the script for default mode: 4')
    mode_of_exec = 4

# If invalid num_datapoints entered, change it to default
if num_datapoints < 1 or num_datapoints > 30:
    num_datapoints = 1

if mode_of_exec == 4:
    data_points = [i for i in range(len(data_json))]
else:
    # Only run for 'medium' type of datapoints
    data_points = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

data_points = data_points[:min(len(data_points), num_datapoints)]

if mode_of_exec == 4 and num_datapoints == 1:
    data_points = [27]

print(f'>>> Mode of execution: {mode_dict[mode_of_exec]}')
print(f'>>> Datapoints: {data_points}')

for i in data_points:
    out = []
    # Unit test generation
    print(f'Starting UNIT TEST GENERATION task for the object #{i + 1}...')

    # Get the meta-data for the current method
    service_name = data_json[i]['serviceName']
    method_qlf_name = data_json[i]['functionQualifiedName']
    status, class_qual_name, class_vars, class_methods, method_params, method_vars = get_metadata(
        service_name, method_qlf_name)
    method_class_name = extract_class_from_class_qlf_name(class_qual_name)

    # Creating a list for meta-data to be used later in the prompts
    meta_data_arr = [method_class_name]
    first_ = {
        "qualifiedName": class_qual_name,
        "variables": class_vars if mode_of_exec == 1 or mode_of_exec == 4 else [],
        "methods": class_methods if mode_of_exec == 1 or mode_of_exec == 4 else []
    }

    meta_data_arr.append(f'{method_class_name}: {json.dumps(first_)}')
    meta_data_arr.append(""
                         if method_params == "" or mode_of_exec in [0, 1, 3] else json.dumps(method_params))
    meta_data_arr.append("" if method_vars == "" or mode_of_exec in [
                         0, 1, 2] else json.dumps(method_vars))

    # Pausing the script for runs other than 1st because of Rate Limiting rules
    if not is_first:
        time.sleep(60)

    # Call to create Unit tests
    tests(i+1, data_json[i], out, meta_data_arr, mode_of_exec)

    is_first = False

print(f'Task Completed')
