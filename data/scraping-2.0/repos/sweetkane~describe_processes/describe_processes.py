#!/opt/homebrew/bin/python3

import os
import subprocess
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

osType = "MacOS"
process_list_path = ".output.txt"
process_description_path = "process_descriptions.csv"

print("Gathering processes...")

subprocess.run(["./get_processes.sh", process_list_path])
subprocess.run(["rm", "-f", process_description_path])

print("Loading processes to memory...")

with open(process_list_path, 'r') as process_list_file:
    process_list = process_list_file.readlines()
process_list = [process.strip() for process in process_list]

subprocess.run(["rm", "-f", process_list_path])

print("Querying LLM for process descriptions...")

messages = [
    {"role": "user", 
     "content": f"""
        I will provide you with a list of {osType} processes. \
        Generate a description for each process, \
        and format the result as a csv with columns process,maker,description.
        The maker column should contain your best guess as to who made the process, \
            i.e. Apple, Microsoft, Google, Cloudflare.
        Each description should be succinct but informative, \
            it should explain exactly what the process does in roughly 10 words, \
            it should not include a comma, \
            and it should also be wrapped in double quotes \
        Do not say anything else, only return the CSV.
        Here are the processes: `{', '.join(process_list)}`
        """
     }
]
# functions = [
#     {
#         "name": "print_process_descriptions",
#         "description": "Given a list of processes and their descriptions, print the list to stdout",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "process_descriptions": {
#                     "type": "object",
#                     "description": "List of objects where list[i] = { \"process\": <description of process> }",
#                 },
#             },
#             "required": ["process_descriptions"],
#         },
#     }
# ]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    # functions=functions,
    # function_call="auto",  # auto is default, but we'll be explicit
)

response_message = response["choices"][0]["message"]["content"]
print(response_message)

with open(process_description_path, 'w') as process_description_file:
    process_description_file.write(response_message)

# if response_message.get("function_call"):
#     function_args = json.loads(response_message["function_call"]["arguments"])
#     descriptions = function_args.get("process_descriptions")
#     print(descriptions)