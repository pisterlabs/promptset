import subprocess
import openai
import os
import json
import sys
sys.path.append('..')
from llm.message_parser import format_source_code_str, parse_source_code_str
from utils.file_io_helper import update_source_code
from utils.github_helper import commit_changes, get_diff_string, DEV_BRANCH_NAME
from utils.error_helper import raise_standard_error

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_code_changes(prompt, source_code_dict):
    source_code_str = format_source_code_str(source_code_dict)
    response = {}

    try:
        response = openai.Edit.create(
            engine="code-davinci-edit-001",
            input=source_code_str,
            instruction=prompt,
            temperature=0,
            top_p=1
        )
    except openai.error.InvalidRequestError as e:
        raise_standard_error(400, "Repo and/or prompt is too large :(")

    generated_code_str = response.choices[0].text
    filenames = list(source_code_dict.keys())
    generated_code_dict = parse_source_code_str(generated_code_str, filenames)
    return generated_code_dict

def generate_diff(prompt, source_code_dict):
    # NOTE: Reset the source code before generating the diff
    update_source_code(source_code_dict)
    generated_code_dict = generate_code_changes(prompt, source_code_dict)
    update_source_code(generated_code_dict)
    commit_changes("Modified based on prompt")

    # TODO: handle when branch is "master" not main
    diff_string = get_diff_string("main", DEV_BRANCH_NAME)
    return diff_string

def generate_validated_diff(prompt, source_code_dict, num_validations=0):
    if os.getenv("NUM_VALIDATIONS"):
        num_validations = int(os.getenv("NUM_VALIDATIONS"))

    if num_validations == 0:
        print(f"Number of validations is 0, skipping validation step.")
        diff_string = generate_diff(prompt, source_code_dict)
        return diff_string

    # NOTE: for num_validations try to generate a diff that is validated by the model with reflection
    # If validation fails all num_validations times, then return empty diff string
    for i in range(num_validations):
        diff_string = generate_diff(prompt, source_code_dict)

        validation_passed = validate_generated_code(prompt, source_code_dict, diff_string)

        if (validation_passed):
            print("Validation passed!!")
            return diff_string
        else:
            print("Validation failed! Retrying ...")

    print(f"All {num_validations} validation attemps failed! Returning empty diff.")
    return ""

def validate_generated_code(prompt, source_code_dict, generated_diff):
    # print(f"source_code_dict: {source_code_dict}")
    # print(f"generated_diff: {generated_diff}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": f"{format_source_code_str(source_code_dict)}"},
            {"role": "user", "content": f"For the prompt: {prompt} this is the diff that GPT came up with: {generated_diff}. Does this look good? Respond yes or no."},
        ],
        temperature=0,
        max_tokens=1000
    )

    assistant_message = response.choices[0].message.content
    print(f"Assistant response: {assistant_message}")
    return 'yes' in assistant_message.lower()
