#%%

import openai
import os
import openai
from RotatingTextFile import RotatingTextFile
openai.organization = "org-HRqLWCACWqb2BZgAXGT6UtE8"
openai.api_key = os.environ["OPENAI_API_KEY"]
# openai.Model.list()
# %%
command = "python test.py"
code_file = "test.py"

# run code file and gather stdout and stderr
import subprocess
def run_and_get_output(command):
    # run command and capture both stdout and stderr
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    # wait for the process to terminate
    stdout, stderr = process.communicate()

    return stdout, stderr


def make_prompt(code_contents, stdout, stderr, command):
    prompt_template = open("prompt.txt", "r").read()
    prompt = prompt_template.format(
        code=code_contents,
        stdout=stdout,
        stderr=stderr,
        command = command,
    )
    return prompt

#%%

def maktlo_iterate(command, code_file):
    stdout, stderr = run_and_get_output(command)
    code_contents = open(code_file, "r").read()

    prompt = make_prompt(code_contents, stdout, stderr, command)

    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=182,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"]
    )
    fixed = response['choices'][0]['text'].strip()

    def condition():
        return True #always rotate

    with RotatingTextFile(code_file,condition,100) as fp:
        fp.write(fixed)
    with RotatingTextFile(code_file+".stdout",condition,100) as fp:
        fp.write(str(stdout))
    with RotatingTextFile(code_file+".stderr",condition,100) as fp:
        fp.write(str(stderr))

# test fixed
for i in range(10):
    maktlo_iterate("python test.py", "test.py")
# %%
