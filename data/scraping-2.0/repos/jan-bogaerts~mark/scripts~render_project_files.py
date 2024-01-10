import sys
import os
from time import sleep
from utils import clean_dir
from constants import  get_model_config, DEFAULT_MAX_TOKENS, OPENAI_API_KEY
import project
import get_project_files
import get_packages_list_file
import json

import openai
import tiktoken

# todo: replace 'MainWindow' with the name of the first component of the project

system_prompt = """Act as a full-stack ai software developer.
We are writing the code for the application called '{0}'.

It uses the following development stack:
{1}
"""
user_prompt = """The following files need to be written:
{0}
Use the component 'MainWindow', located in the file './mainWindow.js' as the root of the app.
only write the code for '{1}'"""
term_prompt = """
Use small functions.
Add documentation to your code.
only write valid code
do not include any intro or explanation, only write code
add css styling

bad response:
```javascript
const a = 1;
```

good response:
const a = 1;
"""


def generate_response(params, key):

    total_tokens = 0
    model = get_model_config('render_component', key)
    
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
    prompt = system_prompt.format(params['app_name'], params['dev_stack'] ) + term_prompt
    messages.append({"role": "system", "content": prompt})
    total_tokens += reportTokens(prompt)
    prompt = user_prompt.format(params['files_desc'], params['file'])
    messages.append({"role": "user", "content": prompt})
    total_tokens += reportTokens(prompt)
    
    total_tokens *= 4  
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


def collect_response(title, response, root_path):
    file_name = title.replace(" > ", "/")
    file_path = os.path.join(root_path, file_name.replace(" ", "_"))
    with open(file_path, "w") as writer:
        writer.write(response)


def process_data(root_path):
    project_desc = project.fragments[0]
    dev_stack = project.fragments[1].content

    project_files = get_project_files.text_fragments[0].data
    package_file = get_packages_list_file.text_fragments[0].content.strip()
    for file in project_files['files']:
        if file != package_file:
            params = {
                'app_name': project_desc.title,
                'files_desc': json.dumps(project_files['explanation']),
                'file': file,
                'dev_stack': dev_stack,
            }
            response = generate_response(params, project_desc.title)
            if response:
                # remove the code block markdown, the 3.5 version wants to add it itself
                response = response.strip() # need to remove the newline at the end
                if response.startswith("```javascript"):
                    response = response[len("```javascript"):]
                if response.endswith("```"):
                    response = response[:-len("```")]
                collect_response(file, response, root_path)
                    
                


def main(prompt, files, package_file, root_path=None):
    # read file from prompt if it ends in a .md filetype
    if prompt.endswith(".md"):
        with open(prompt, "r") as promptfile:
            prompt = promptfile.read()

    print("loading project")

    # split the prompt into a toolbar, list of components and a list of services, based on the markdown headers
    project.split_standard(prompt)
    get_project_files.load_results(files)
    get_packages_list_file.load_results(package_file)

    print("rendering results")

    # save there result to a file while rendering.
    if root_path is None:
        root_path = './'
    
    process_data(root_path)
    
    print("done! check out the output file for the results!")


text_fragments = []  # the list of text fragments representing all the results that were rendered.

def load_results(filename):
    with open(filename, "r") as reader:
        current_content = ''
        current_title = ''
        for line in reader.readlines():
            if line.startswith('#'):
                if current_title != '':
                    text_fragments.append(project.TextFragment(current_title, current_content))
                current_title = line
                current_content = ''
            else:
                current_content += line
        if current_title != '':
            text_fragments.append(project.TextFragment(current_title, current_content))
        

if __name__ == "__main__":

    # Check for arguments
    if len(sys.argv) < 4:
        print("Please provide a prompt and a file containing the components to check")
        sys.exit(1)
    else:
        # Set prompt to the first argument
        prompt = sys.argv[1]
        files = sys.argv[2]
        package_file = sys.argv[3]

    # Pull everything else as normal
    file = sys.argv[4] if len(sys.argv) > 4 else None

    # Run the main function
    main(prompt, files, package_file, file)
