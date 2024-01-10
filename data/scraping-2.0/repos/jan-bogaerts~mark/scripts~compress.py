import sys
import os
from time import sleep
from utils import clean_dir
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import result_loader
import constant_lister

import openai
import tiktoken


ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """Act as an ai software analyst. It is your job to shorten the following text as much as possible and rephrase it in your own words, without loosing any meaning.
Remove the markdown, but use bullet points where appropriate.
compress the following text:
"""
user_prompt = """{0}"""
term_prompt = ""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('compres', key)
    
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
            + prompt[:100]
            + "\033[0m"
        )
        return token_len

    # Set up your OpenAI API credentials
    openai.api_key = OPENAI_API_KEY

    messages = []
    prompt = system_prompt.format()
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['feature_description'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "system", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens *= 2  # max result can be as long as the input, also need to include the input itself
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

    for to_check in project.fragments:  # skip the first two fragments cause that's the description and dev stack
        if not to_check.content: # skip empty fragments
            continue
        if ONLY_MISSING and has_fragment(to_check.full_title):
            continue
        content = constant_lister.get_fragment(to_check.full_title, to_check.content)
        params = {
            'feature_description': content,
        }
        response = generate_response(params, to_check.full_title)
        if response:
            collect_response(to_check.full_title, response, result, writer)
    return result
                    


def main(prompt, consts, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()


    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    constant_lister.load_results(consts)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
    
    file_name = file + "_compress.md"
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


def has_fragment(title):
    '''returns true if the title is in the list of fragments'''
    to_search = title.strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title == to_search:
            return True
    return False


def get_fragment(full_title):
    to_search = full_title.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.full_title.lower().strip() == to_search:
            return fragment
    return None


if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 3:

        # Looks like we don't have a prompt. Check if prompt.md exists
        if not os.path.exists("prompt.md"):

            # Still no? Then we can't continue
            print("Please provide a prompt")
            sys.exit(1)

        # Still here? Assign the prompt file name to prompt
        prompt = "prompt.md"

    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        consts = sys.argv[2] 

    # Pull everything else as normal
    file = sys.argv[3] if len(sys.argv) > 3 else None

    # Run the main function
    main(prompt, consts, file)
