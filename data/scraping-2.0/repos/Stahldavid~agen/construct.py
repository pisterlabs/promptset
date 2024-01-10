
from definitions import *
import os
from dotenv import load_dotenv
import openai
import json
from functions_ca import functions1, functions2, functions3, functions4

import json
import openai
from code_search import similarity_search



query = "Variable impedance control for force feedback"
results_string = similarity_search(query)
print(results_string)


import openai
import os
import json
import time

def auto_code(query):

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "content": "You are 'CodeMaster', a virtual architect specializing in Python. Your mission: decompose an intricate programming challenge into a structured roadmap of sub-tasks. For each sub-task, you will craft an individual Python file. These files should not only adhere to PEP8 standards but also be written in a manner that even a novice programmer could understand. The metadata of each file must be comprehensive, detailing its name, its primary purpose, an index of functions or classes, and its interdependencies with other files. Remember, one of these files should be designated as the 'main'—the cornerstone of this project, tying all the other modules together. Approach this as if you are crafting a masterpiece for a prestigious coding competition."
            }
            ,
            {
                "role": "user",
                "content": f"Given the complex code generation task: {query} using ros2, webots, webots_ros2 and ros2_control.', please decompose it into a detailed, numbered list of sub-tasks. Each sub-task should be a independent file should contain the name of the python file, description, all the functions and classes from the file, as well releted files. Make sure to devide the task into minimum 5 files. Try to make the code as readable as possible, encapsulating the code into functions and classes. Lets think step by step.\n\nThe following are the retrived documents from all the relevant repositories based on the query :\n{query}\nThese retrived functions from all the relevant repositories are helpfull but not fullfile the user task. Please use this context to help guide the task decomposition"
            }
        ],
        functions=functions1, 
        function_call={"name": "generate_code"}
    )

    reply_content = completion.choices[0]
    args = reply_content["message"]['function_call']['arguments']
    data = json.loads(args)
    files = {}
    for file in data["files"]:
        files[file["code_blocks"][0]["name"]] = {
            "order": file["order"],
            "code_blocks": file["code_blocks"],
        }
    files = dict(sorted(files.items(), key=lambda item: item[1]['order']))
    files_string = ""
    for filename, file_data in files.items():
        files_string += f"Order of development: {file_data['order']}\n"
        files_string += f"{filename}:\n"
        for block in file_data['code_blocks']:
            files_string += f"  Code block type: {block['type']}\n"
            files_string += f"  Code block name: {block['name']}\n"
            files_string += f"  Code block description: {block['description']}\n"
            files_string += f"  Code block content: {block['content']}\n"

    completion2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are an advanced AI with capabilities to analyze intricate code and pseudocode files. Based on this analysis, you provide recommendations for the most appropriate vectorstore repositories to extract relevant code snippets from. In addition, you generate search queries that could be utilized to fetch these helpful code samples."
            },
            {
                "role": "user",
                "content": f"I need your expertise to examine the provided code and pseudocode files. Your task is to pinpoint any issues, inefficiencies, and areas for potential enhancements. Here are the files you need to look into:\n\n{files_string}"
            }
        ],
        functions = functions2,
        function_call={"name": "analyze_code"}
    )
    reply_content2 = completion2.choices[0]
    args2 = reply_content2["message"]['function_call']['arguments']
    data2 = json.loads(args2)

    directories = ['db_ros2_control', 'db_ros2', 'db_webots_ros2', 'db_webots']
    results = {}
    for file_data in data2["files"]:
        file_name = file_data["file_name"]
        repository = file_data["repository"]
        query = repository['query']
        result = similarity_search(query, directories)
        results[f"{file_name}_{query}"] = result

    file_strings = {}
    for filename, file_data in files.items():
        file_lines = []
        file_lines.append(f"Order of development: {file_data['order']}")
        file_lines.append(f"{filename}:")
        for block in file_data['code_blocks']:
            file_lines.append(f"  Code block type: {block['type']}")
            file_lines.append(f"  Code block name: {block['name']}")
            file_lines.append(f"  Code block description: {block['description']}")
            file_lines.append(f"  Code block content: {block['content']}")
            for key, value in results.items():
                if filename in key:
                    file_lines.append(f"#####################################")
                    file_lines.append(f"  Query:\n\n {key.replace(filename+'_', '')}")
                    file_lines.append(f"\n\n  Query result:\n\n {value}")
        file_strings[filename] = '\n'.join(file_lines)

    output_string = ""
    new_files = {}
    new_comments = {}
    target_dir = "/home/stahlubuntu/coder_agent/bd"
    for filename, file_string in file_strings.items():
        wait_time = 1
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            try:
                completion4 = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI Code Optimization Model that can optimize code, complete #TODOs, recommend best practices, and learn from relevant code repositories. Your main task is to analyze the given code, which performs semantic search queries on a vectorstore using OpenAI embeddings, and apply the insights from the search results to refine the code. The final goal is to produce a fully functional and optimized code file that can be used as part of a larger project, specifically a variable impedance control code for force feedback"
                        },
                        {
                            "role": "user",
                            "content": f"I am working on a coding project that aims to develop a variable impedance control code for force feedback. I need your expertise to improve my code. Here is the current version of one file of my code, along with the semantic search queries I’ve done on a vectorstore using OpenAI embeddings, and the results of these queries:\n\n{file_string}\n\nCan you improve this code, using the suggestions from the semantic search results? Please write the improved and complete code file. Please complete and improve the file based on the context."
                        }
                    ],
                    functions=functions4,
                    function_call={"name": "optimize_code"}
                )

                reply_content4 = completion4.choices[0]
                args4 = reply_content4["message"]['function_call']['arguments']
                data4 = json.loads(args4)

                new_files[filename] = data4['code']
                new_comments[filename] = data4['comments']
                output_string += f"For file: {filename}, the improved code is: {new_files[filename]}\n"
                break
            except openai.error.OpenAIError as e:
                time.sleep(wait_time)
                attempts += 1
                wait_time *= 2

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename, code_content in new_files.items():
        comments_content = new_comments.get(filename, '')
        code_file_path = os.path.join(target_dir, filename)
        comments_file_path = os.path.join(target_dir, f"{filename}_comments.txt")
        with open(code_file_path, "w") as code_file:
            code_file.write(code_content)
        with open(comments_file_path, "w") as comments_file:
            comments_file.write(comments_content)

    return output_string

# To use the function:
# result = auto_code(query)
# print(result)
