import json
import os

import openai

from brain.dict_2_array import dict_2_array
from brain.dict_comparator import return_non_matching_values
from brain.tiktoken_counter import count_total_tokens, count_num_tokens_price
from brain.token_counter import count_tokens
from eyes.read_file import read_files_and_hash, read_json_file
from hands.get_image import get_full_from_relative
from hands.reformat_2_pep8 import replace_string_lf_with_crlf
from tools.git_patcher import get_patch
from tools.logger import logger, logger_err

engine = "text-davinci-003"


def generate_refactored_code(file_contents):
    # Generate responses for each file content with increasing prompts
    responses = {}
    try:
        for file_hash, file_content_tuple in file_contents.items():
            file_content = file_content_tuple[0]
            file_name = file_content_tuple[1]
            prompt = "Refactor the given Python code to adhere to PEP 8 guidelines. " \
                     "Do not write comments in the code ever.\n" \
                     "If you find comments like this '# Flaw: ' followed by text, " \
                     "assume they are recommendations on how to fix the code, you should " \
                     "follow them if possible and remove them\n" \
                     "Code:" \
                     " \n\n```" + replace_string_lf_with_crlf(file_content) + "```\n"
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=count_tokens(prompt) * 2,
                temperature=0,
                # model="gpt-3.5-turbo-8",  # 8K model
                n=1
            )
            logger("Usage of the request:\n" + str(response.usage))
            responses[file_name] = response.choices[0].text.strip()
    except Exception as e:
        logger_err(e)
    return responses


def output_responses(responses, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # Output the resulting files with the responses
    for file_name, response in responses.items():
        base_name, extension = os.path.splitext(file_name)
        if ".rfct" in base_name:  # If it's already a rfct file, don't add the extension
            output_file = os.path.join(output_folder, base_name + extension)
        else:
            output_file = os.path.join(output_folder, base_name + ".rfct" + extension)
        with open(output_file, 'w') as file:
            file.write(response)


def refactor_files(target_file_contents):
    # Generate the refactored code from non-matching values
    refactored_codes = generate_refactored_code(target_file_contents)
    # Create the ".ai_refactored" folder in the same directory as the target folder
    output_folder = os.path.join(to_be_refactored_folder, ".ai_refactored")
    # Output the refactored code
    output_responses(refactored_codes, output_folder)
    # Update the long term memory
    output_content = read_files_and_hash(output_folder)
    file_contents_json = json.dumps(output_content)  # Convert dictionary to JSON string
    # Write the long term memory to the hidden folder
    os.makedirs(hidden_folder_path, exist_ok=True)
    with open(hidden_file_path, "w") as file:
        file.write(file_contents_json)


def refactor_destination(folder_path):
    global hidden_folder_path, hidden_file_path, to_be_refactored_folder, non_matching_values  # TODO: remove globals
    # Read long term memory, to see which files have been refactored already
    hidden_folder_path = get_full_from_relative("../.cartuli")
    hidden_file_path = os.path.join(hidden_folder_path, "long_term_hash_memory.json")
    long_term_hash_memory = read_json_file(hidden_file_path)
    # Read the target folder
    to_be_refactored_folder = get_full_from_relative(folder_path)
    target_file_contents = read_files_and_hash(to_be_refactored_folder)
    # Compare the two dictionaries and get only the non-matching values
    non_matching_values = return_non_matching_values(long_term_hash_memory, target_file_contents)

    total_tokens_code = count_total_tokens(non_matching_values)
    total_tokens_code_price = count_num_tokens_price(total_tokens_code, engine)
    if len(non_matching_values) > 0:
        logger("Non matching values:\n" + json.dumps(non_matching_values))
    if non_matching_values is not None and \
            len(non_matching_values) > 0 or \
            len(long_term_hash_memory) == 0:
        refactor_and_patch(non_matching_values, target_file_contents, to_be_refactored_folder)
    else:
        user_input = input("No new files to refactor, do you want to do the refactor anyway? y/n: ")
        if user_input.lower() == "y":
            refactor_and_patch(non_matching_values, target_file_contents, to_be_refactored_folder)


def refactor_and_patch(non_matching_values, target_file_contents, to_be_refactored_folder):
    refactor_files(non_matching_values or target_file_contents)
    file_array = dict_2_array(non_matching_values or target_file_contents)
    get_patch(to_be_refactored_folder, file_array)


if __name__ == "__main__":
    refactor_destination("../code_to_be_refactored")
