import os
import openai

from brain.token_counter import count_tokens
from mouth.asker import get_open_ai_key


def read_files(folder_path):
    # Read all files in the given folder
    file_contents = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                file_contents[file_name] = file.read()
    return file_contents


def generate_responses(file_contents):
    # Set up OpenAI API credentials
    openai.api_key = get_open_ai_key()

    # Generate responses for each file content with increasing prompts
    responses = {}
    for file_name, content in file_contents.items():
        prompt = "Refactor this code as a Senior Engineer, avoid shadowing " \
                 "variables if possible, write code with the minimum amount of " \
                 "python warnings and with correct python default indentation" \
                 " ``` " + content + "```"
        token_number = count_tokens(prompt)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=token_number,
            # top_p=0.2,
            temperature=0,
            n=1
        )
        responses[file_name] = response.choices[0].text
    return responses


def output_responses(responses, folder_path):
    # Create the "generated" folder
    output_folder = os.path.join(folder_path, "ai_refactored")
    os.makedirs(output_folder, exist_ok=True)

    # Output the resulting files with the responses
    for file_name, response in responses.items():
        base_name, extension = os.path.splitext(file_name)
        output_file = os.path.join(output_folder, base_name + ".rfct" + extension)
        with open(output_file, 'w') as file:
            file.write(response.strip())


# Main script
folder_path = "C:/Users/sergi/PycharmProjects/Cartuli-AI-Assitant/scripts/python/code_to_be_refactored"
file_contents = read_files(folder_path)
responses = generate_responses(file_contents)
output_responses(responses, folder_path)
