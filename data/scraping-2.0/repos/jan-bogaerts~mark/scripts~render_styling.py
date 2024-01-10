import sys
import os
from time import sleep
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import result_loader
import get_styling_names
import json

import openai
import tiktoken


ONLY_MISSING = True # only check if the fragment has not yet been processed

system_prompt = """using the following development stack:
{0}
write out all the styling classes:
{1}

based on the user's text
only return styling, no code. Don't return any explanation or introduction or editor formatting."""
user_prompt = "{0}"
term_prompt = """remember: only return properly formatted css styling, no code, no explanation, no introduction."""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('render_styling', key)
    
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
    prompt = system_prompt.format(params['dev_stack'], params['styling_names'])
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['feature_description'])
    messages.append({"role": "user", "content": prompt} )
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    total_tokens = DEFAULT_MAX_TOKENS  # max result needs to be short
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


def collect_file_list(title, file_names, writer):
    names_str = json.dumps(file_names)
    writer.write(f'# {title}\n')
    writer.write(f'{names_str}\n')
    writer.flush()


def collect_response(filename, response):
    file_name = filename.replace(".js", ".css")
    with open(file_name, "w") as writer:
        writer.write(response)
    return file_name



def process_data(writer):
    result = []
    dev_stack = project.fragments[1].content
    for fragment in project.fragments:
        if ONLY_MISSING and has_fragment(fragment.full_title):
            continue
        files = get_styling_names.get_data(fragment.full_title)
        for file_name, styling_names in files.items():    
            if len(styling_names):
                file_names = [] # keep track of the file names generated for this fragment, so we can save it in the markdown file
                response = generate_response({
                    'styling_names': '- ' + '\n- '.join(styling_names),
                    'dev_stack': dev_stack,
                    'feature_description': fragment.content
                }, fragment.full_title)
                if response:
                    # remove the code block markdown, the 3.5 version wants to add it itself
                    response = response.strip() # need to remove the newline at the end
                    if response.startswith("```css"):
                        response = response[len("```css"):]
                    if response.endswith("```"):
                        response = response[:-len("```")]
                    file_name = collect_response(file_name, response)
                    file_names.append(file_name)
                if file_names:
                    collect_file_list(fragment.full_title, file_names, writer)
            
    return result
                    


def main(prompt, styling_list, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    get_styling_names.load_results(styling_list)

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
    
    file_name = file + "_styling_files.md"
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


if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 3:
        print("Please provide a prompt and a file containing the components to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        styling_names = sys.argv[2]

    # Pull everything else as normal
    file = sys.argv[3] if len(sys.argv) > 3 else None

    # Run the main function
    main(prompt, styling_names, file)
