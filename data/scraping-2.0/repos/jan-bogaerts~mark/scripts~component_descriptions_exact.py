"""
Create a more detailed description of the components that are declared in a text fragment.
first run the component_lister.py and compress.py scripts to get the list of components.
"""

import sys
import os
import json
from time import sleep
from constants import  get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import compress
import component_lister
import primary_component
import result_loader


import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """Act as an ai feature classification system.
It is your task to build the feature list related to the UI component '{0}' using the provided feature list.

Only return what is in the feature list about {0}{1}. No introduction or explanation."""
user_prompt = """The feature list:
{0}"""
term_prompt = """{0}"""


def generate_response(params, key):

    total_tokens = 0
    key = key.split('#')[-1].strip()
    model = get_model_config('component_descriptions_exact', key)
    
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
    prompt = system_prompt.format(params['class_name'], params['other_classes'] ) 
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['feature_description'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    term_prompt = params['remember_prompt']
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens += int(total_tokens /2)
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


def add_result(to_add, writer):
    writer.write(to_add + "\n")
    writer.flush()


def collect_response(title, response, writer):
    # title comes from the text fragment, already has the # in it
    add_result(title, writer)
    add_result(response, writer)


def process_data(writer):
    for fragment in component_lister.text_fragments:
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        result = {}
        # process each key-value pair of the json data structure
        components = fragment.data
        primary = primary_component.get_primary(fragment.full_title)
        for comp_name in components:
            comp_info = compress.get_fragment(fragment.full_title)
            other_components = [c for c in components if c != comp_name]

            remember_prompt = ''
            if len(other_components) == 1:
                to_be = ' is'
                other_components = other_components[0]
            elif len(other_components) > 1:
                to_be = ' are'
                if comp_name == primary:
                    last_join = ' and '
                else:
                    last_join = ' or '
                other_components = ', '.join(other_components[:-1]) + last_join + other_components[-1]
            else:
                other_components = ''
            if other_components:
                if comp_name == primary:
                    remember_prompt = 'Remember: mention where ' + other_components + to_be + ' used but not their features'
                    other_components = ', only mention where ' + other_components + to_be + ' used but not their features'
                else:
                    other_components = ', nothing about ' + other_components


            if comp_info:
                params = {
                    'class_name': comp_name,
                    'other_classes': other_components,
                    'feature_description': comp_info.content,
                    'remember_prompt': remember_prompt
                }
                response = generate_response(params, fragment.full_title)
                if response:
                    result[comp_name] = response
        collect_response(fragment.full_title, json.dumps(result), writer)



def main(prompt, compressed, list, primary, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers

    compress.load_results(compressed)
    component_lister.load_results(list)
    primary_component.load_results(primary)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
        
    open_mode = 'w'
    filename= file + "_comp_descriptions_exact.md"
    if ONLY_MISSING:
        load_results(filename)
        open_mode = 'a'

    print("rendering results")
    
    with open(filename, open_mode) as writer:
        process_data(writer)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.

def load_results(filename, overwrite_file_name=None, overwrite=True):
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    result_loader.load(filename, text_fragments, True, overwrite_file_name)
        

def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False


def get_data(title):
    '''returns the list of components for the given title'''
    to_search = title.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title.lower() == to_search:
            return fragment.data or []
    return []    


def get_description(title, component):
    data = get_data(title)
    if component in data:
        return data[component]
    return None

if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 5:
        print("Please provide a prompt and a file containing the components to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        compressed = sys.argv[2]
        declare_or_use_list = sys.argv[3]
        primary = sys.argv[4]

    # Pull everything else as normal
    file = sys.argv[5] if len(sys.argv) > 5 else None

    # Run the main function
    main(prompt, compressed, declare_or_use_list, primary, file)
