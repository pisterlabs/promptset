import sys
import os
from time import sleep
from utils import clean_dir
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import constant_lister
import class_lister
import result_loader

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed


system_prompt = """Act as an ai software analyst.
It is your task to classify which of the following classes is the primary / root class described in the feature list.
only return the name of the class that is the primary / root, no explanation or any more text.
"""
user_prompt = """classes:
{0}

feature list:
# {1}
{2}"""
term_prompt = """Remember: only return the name of the class that is the primary / root class.

good response:
x

bad response:
the primary / root class is x"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('primary_class', key)
    
    def reportTokens(prompt):
        encoding = tiktoken.encoding_for_model(model)
        # print number of tokens in light gray, with first 10 characters of prompt in green
        token_len = len(encoding.encode(prompt))
        print(
            "\033[37m"
            + str(token_len)
            + " tokens\033[0m"
            + " in prompt: "
            + "\033[92m"
            + prompt
            + "\033[0m"
        )
        return token_len

    # Set up your OpenAI API credentials
    openai.api_key = OPENAI_API_KEY

    messages = []
    prompt = system_prompt
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['classes'], params['feature_title'], params['feature_description'])
    messages.append({"role": "user", "content": prompt} )
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens += 20  # max result needs to be short
    if total_tokens > DEFAULT_MAX_TOKENS:
        total_tokens = DEFAULT_MAX_TOKENS
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": total_tokens,
        "temperature": 0,
    }

    # Send the API request
    keep_trying = True
    response = None
    while keep_trying:
        try:
            response = openai.ChatCompletion.create(**params)
            keep_trying = False
        except Exception as e:
            # e.g. when the API is too busy, we don't want to fail everything
            print("Failed to generate response (retrying in 30 sec). Error: ", e)
            sleep(30)
            print("Retrying...")

    # Get the reply from the API response
    if response:
        reply = response.choices[0]["message"]["content"] # type: ignore
        print("response: ", reply)
        return reply
    return None


def add_result(to_add, result, writer):
    result.append(to_add)
    writer.write(to_add + "\n")
    writer.flush()


def collect_response(title, response, result, writer):
    # get the first line in the component without the ## and the #
    add_result(f'# {title}', result, writer)
    add_result(response, result, writer)


def process_data(writer):
    result = []
    for fragment in project.fragments:
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        classes = class_lister.get_classes(fragment.full_title)
        if len(classes) == 1:
            collect_response(fragment.full_title, classes[0], result, writer)
        elif len(classes) > 0:
            content = constant_lister.get_fragment(fragment.full_title, fragment.content)
            params = {
                'classes': ', '.join(classes),
                'feature_title': fragment.title,
                'feature_description': content
            }
            response = generate_response(params, fragment.full_title)
            if response:
                collect_response(fragment.full_title, response, result, writer)
            
    return result
                    


def main(prompt, classes_list, constants, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of classes and a list of services, based on the markdown headers
    project.split_standard(prompt)
    class_lister.load_results(classes_list)
    constant_lister.load_results(constants)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
    
    file_name = file + "_primary_class.md"

    open_mode = 'w'
    if ONLY_MISSING:
        load_results(file_name)
        open_mode = 'a'

    print("rendering results")

    with open(file_name, open_mode) as writer:
        process_data(writer)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.

def load_results(filename, overwrite_file_name=None, overwrite=True):
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    result_loader.load(filename, text_fragments, False, overwrite_file_name)


def get_primary(title):
    '''returns the list of classes for the given title'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return fragment.content
    return []


def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False

if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 4:
        print("Please provide a prompt and a file containing the classes to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        classes_list = sys.argv[2]
        constants = sys.argv[3]

    # Pull everything else as normal
    file = sys.argv[4] if len(sys.argv) > 4 else None

    # Run the main function
    main(prompt, classes_list, constants, file)
