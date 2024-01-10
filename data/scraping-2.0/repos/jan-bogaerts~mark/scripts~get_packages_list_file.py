import sys
from time import sleep
from constants import get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import result_loader
import get_project_files

import openai
import tiktoken


# cant ONLY_MISSING = False # only check if the fragment has not yet been processed

system_prompt = """Act as a full-stack ai software developer.
We are writing the code for the application called '{0}'.

Which of the following files contains the list with all the required packages?

Only return the name of the file, no explanation.
"""
user_prompt = """{0}"""
term_prompt = """"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('get_package_list_file', key)
    
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
    prompt = system_prompt.format(params['project_name']) #+ term_prompt
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['file_list'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    if term_prompt:
        messages.append({"role": "assistant", "content": term_prompt})
        total_tokens += reportTokens(term_prompt)
    
    
    total_tokens += 50  # only want 1 filename, so  this should be enough
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


def process_data(writer):
    project_desc = project.fragments[0]
    list = get_project_files.text_fragments[0].data['explanation']

    params = {
            'project_name': project_desc.title,
            'file_list': list,
        }
    response = generate_response(params, project_desc.title)
    if response:
        collect_response(project_desc.title, response, writer)



def main(prompt, project_files, file=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    get_project_files.load_results(project_files)

    print("rendering results")

    # save there result to a file while rendering.
    if file is None:
        file = 'output'
    with open(file + "_packages_file.md", "w") as writer:
        process_data(writer)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.

def load_results(filename, overwrite_file_name=None, overwrite=True):
    if not overwrite_file_name and overwrite:
        # modify the filename so that the filename without extension ends on _overwrite
        overwrite_file_name = filename.split('.')[0] + '_overwrite.' + filename.split('.')[1]
    result_loader.load(filename, text_fragments, False, overwrite_file_name)
    

def get_data(title):
    '''returns the list of components for the given title'''
    to_search = title.lower().strip()
    if not to_search.startswith('# '):
        to_search = '# ' + to_search
    for fragment in text_fragments:
        if fragment.title.lower() == to_search:
            return fragment.data or []
    return []    

if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 3:
        print("Please provide a prompt")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        project_files = sys.argv[2] 

    # Pull everything else as normal
    file = sys.argv[3] if len(sys.argv) > 3 else None

    # Run the main function
    main(prompt, project_files, file)
