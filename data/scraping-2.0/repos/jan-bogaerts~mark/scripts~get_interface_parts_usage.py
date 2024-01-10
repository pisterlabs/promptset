# note: special case of prompt: called by the render-component.py & render-service scripts to describe which parts of an interface was used
import sys
import os
from time import sleep
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
# import render_component
import json
import result_loader

import openai
import tiktoken

ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """This is the known interface for {0}:
{1}
List everything from the known interface described above that is used in the following feature description.

Return the result as a json object of key-value pairs where the value is a short description of the key with enough information so that a component can use the feature in code. Do not include by who the key is used in the description. Do not include any introduction or explanation. Return an empty object if nothing is found."""
user_prompt = """feature description:
{0}"""
term_prompt = "Remember: only include items from the known interface that are required for the feature descriptions"


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('get_interface_parts_usage', key)
    
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
    prompt = system_prompt.format(params['interface'], params['interface_content'])
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['content'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens *= 2 
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
    # get the first line in the component without the ## and the #
    add_result(f'# {title}', writer)
    add_result(response, writer)




text_fragments = []  # the list of text fragments representing all the results that were rendered.
loaded_from_filename = None

def load_results(filename, overwrite_file_name=None, overwrite=True):
    global loaded_from_filename
    loaded_from_filename = filename
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    if os.path.exists(filename):
        result_loader.load(filename, text_fragments, True, overwrite_file_name)
        

def save_results():
    if not loaded_from_filename:
        print('no file loaded yet for the interface_parts')
        raise Exception('no file loaded yet for the interface_parts')
    with open(loaded_from_filename, 'w') as writer:
        for fragment in text_fragments:
            collect_response(fragment.title.split('# ')[-1], json.dumps(fragment.data), writer)


def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False    


def get_fragment(title):
    '''returns the fragment for the given title'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return fragment
    return None


def get_data(title):
    '''returns the list of components for the given title'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return fragment.data or []
    return []


def get_interface_parts(title, interface_name):
    to_search = title.strip()
    search_for_interface = interface_name.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    result = {}
    for fragment in text_fragments:
        if to_search in fragment.data:
            section = fragment.data[to_search]
            # doing case insensitive search here, otherwise things can go wrong
            temp_section = {k.lower(): v for k, v in section.items()}
            if search_for_interface in temp_section:
                for key, value in temp_section[search_for_interface].items():
                    result[key] = value
    return result

def list_used_interface_parts(interface_name, interface_title, interface_content, content, content_title):
    '''lists all the parts of an interface for a service that are used in a component'''
    content_title = content_title.strip()
    if not content_title.startswith('# '):
        content_title = '# ' + content_title
    if not interface_title.startswith('# '):
        interface_title = '# ' + interface_title
    fragment = get_fragment(content_title)
    if not fragment:
        fragment = project.TextFragment(content_title, '')
        fragment.data = {}
        text_fragments.append(fragment)
    if not fragment.data:
        fragment.data = {}

    # store each interface together with where it's declared, so we can reconstruct correctly again later on.
    if not interface_title in fragment.data:
        fragment.data[interface_title] = {} 
    interface_data = fragment.data[interface_title]  
    
    params = {
        'content': content,
        'interface': interface_name,
        'interface_content': interface_content,
    }
    response = generate_response(params, content_title) # using the code title here, cause that one determines the complexity of the code
    if response:
        try:
            response = json.loads(response)
        except:
            print(f'failed to parse json: {response}')
            response = {}
        if not interface_name in interface_data:
            interface_data[interface_name] = response
        else:
            already_known_int = interface_data[interface_name]
            for key, value in response.items():
                already_known_int[key] = value
        save_results()
        return response



if __name__ == "__main__":

    # Check for arguments
    print('todo')
