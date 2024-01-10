import os
import re
import openai
from dotenv import load_dotenv

load_dotenv()

# CHATGPT API

openai.api_key = os.getenv('API_KEY') # secret key!

def get_chatgpt_alternate(code):
    """ 
    Sends a request to create an alternate of the given code.

    ### Params
    code : str - the cleaned, one-line contents of the code file as a string

    ### Returns
    A string containing the contents of the refactored code, with comments cleaned
    """
    # sending request!
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=1.5,
        max_tokens=3000,
        messages=[
            {"role": "user", "content": f"Refactor the following C code without explanation and MAKE SURE TO write !!!!! every place a newline character should appear: {code}"}
        ]
    )

    # structure of response found at https://platform.openai.com/docs/guides/chat/introduction
    response = completion.choices[0].message.content
    # clean comments, newlines, etc.
    response = re.sub(r'\/\*[\s\S]*?\*\/', '', response)
    response = re.sub(r'//.*\n', '\n', response)
    response = re.sub(r'\s+', ' ', response).strip()

    return response


# FILE SYSTEM

def make_alternates(original_dir_path, new_dir_path, num_alternates):
    """ 
    Iterates through all code files in given directory, generates alternates
    from ChatGPT, and puts these new files into the new directory.

    ### Params
    original_dir_path : str - path to directory of the original code files
    new_dir_path : str - path to directory for placing alternates
    num_alternates: int - the number of alternate files to generate

    ### Returns
    Nothing! Just puts the alternates into the new directory
    """
    # iterate through all files in directory
    i = 0
    for file_name in sorted(list(os.listdir(original_dir_path))):
        original_path = f"{original_dir_path}/{file_name}" # path of that file
        print("ORIGINAL PATH", original_path)

        if file_name == '.DS_Store':
            continue

        with open(original_path, 'r') as f:
            lines = f.read() # read the contents of the file into lines

        print(lines)
        if len(lines) > 1250:
            continue
        # generate `num_alternates` alternate files
        for alt in range(num_alternates):
            # get new file name and path. adds "alt[x]" after name for each alt
            new_file_name = "[ALT] " + file_name.split('.')[0] + f"-{alt}" + ".txt"
            new_path = f"{new_dir_path}/{new_file_name}"

            # call chatgpt api to get alternate
            response = get_chatgpt_alternate(lines)
            print(response)
            with open(new_path, "w") as g:
                g.write('\n'.join(response.split('!!!!!')))

        i += 1
        if i > 25:
            assert False

if __name__ == "__main__":
    # current_file_path = os.path.dirname(os.path.realpath(__file__)) # path of this directory
    # original_dir_path = f"{current_file_path}/uncompiled" # path to get code files from
    # new_dir_path = f"{current_file_path}/chatgpt_alternates" # path to write alternates to
    original_dir_path = 'data/leetcode_data_FINAL/C_COMPILED_FILES'
    new_dir_path = 'data/chatgpt_alternates'
    num_alternates = 3 # number of alternate files for chatgpt to generate
    make_alternates(original_dir_path, new_dir_path, num_alternates)