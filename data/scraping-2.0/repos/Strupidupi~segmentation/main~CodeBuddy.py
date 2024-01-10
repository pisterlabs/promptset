import os
import openai


def read_files_from_path(path):
    file_contents = {}
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as fileHandle:
                file_contents[file_path] = fileHandle.read()
    return file_contents


def concatenate_contents(file_contents):
    concatenated_string = ""
    for path, content in file_contents.items():
        concatenated_string += f"--{path}--\n{content}\n"
    return concatenated_string


def send_request_to_openai(codebase_summary, task):
    openai.api_key = 'YOUR_API_KEY'
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a software developer"},
            {"role": "user", "content": task + "\n" + codebase_summary}
        ]
    )
    return response.choices[0].message.content.strip()


def process_response_and_overwrite_files(response):
    file_blocks = response.split("--")
    for block in file_blocks:
        if block.strip():
            path, content = block.split("--", 1)
            with open(path.strip(), 'w', encoding='utf-8') as file:
                file.write(content.strip())


codebase_path = "path_to_your_directory"
change_task = ""

summary = concatenate_contents(read_files_from_path(codebase_path))
llm_response = send_request_to_openai(summary, change_task)
process_response_and_overwrite_files(llm_response)
