from codecs import ignore_errors
import json
import pprint
import uuid
import os

from datetime import datetime

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request,
)

LOG = ForgeLogger(__name__)

import openai

import pandas as pd
import csv
import traceback
import requests
from bs4 import BeautifulSoup

from forge.chatgpt import ChatGPT
from forge.executor import Executor 

# class Executor:

#     def run(self, code, globals_env=None, locals_env=None):
#         try:
#             tmp_code = ""
#             for line in code.split("\n"):
#                 if not line.startswith("```"):
#                     tmp_code += line + "\n"

#             exec(tmp_code, globals_env, locals_env)
#         except Exception as e:
#             return str(traceback.format_exc())[-100:]
#         return None


prompt_create_file = """Your task is:

{}

INSTRUCTIONS

Use python code to solve the task. You can use pandas package if needed. Load data with proper delimiter separator.

Please start the file with `# new_file filename.py`. 

Please call the function that is solving the task if needed.

Please print the final response.

Please return python code only.
""" 

prompt_run_python = """Your task is:

{}

Please return python code only that solves the task.
""" 


from forge.data_engineer.utils import files_to_prompt


class DataEngineerAgent:
    def __init__(self, task):
        self.task = task
        
        self.chat = ChatGPT()
        self.steps = [] 

    def run(self):
        wd = os.getcwd()
        files = [f for f in os.listdir(wd) if os.path.isfile(os.path.join(wd, f))]
        
        LOG.info("Input files")
        LOG.info(str(files))

        inc_files = files_to_prompt(wd, files)

        task_prompt = self.task
        if inc_files != "":
            task_prompt += f"\n\nThere are {len(files)} files in the current directory. Files are listed below:\n\n"
            task_prompt += inc_files

        prompt = prompt_create_file.format(task_prompt)

        LOG.info(f"Prompt: {prompt}")
        response = self.chat.chat(prompt)
        LOG.info(f"Response: {response}")
    
        code = response
        tmp_code = ""
        code_fname = ""
        
        for i, line in enumerate(code.split("\n")):
            if i == 0 and line.startswith("#"):
                code_fname = line.split(" ")[-1]
            if not line.startswith("```"):
                tmp_code += line + "\n"

        LOG.info(f"Create file {code_fname}")
        if code_fname != "":
            with open(code_fname, "w") as fin:
                fin.write(tmp_code)

        output = ""
        # execute if there is not __name__ == __main__
        if "if __name__ ==" not in tmp_code:

            executor = Executor()
            output, error = executor.run(tmp_code, globals(), locals())
            if error is not None and error != "":
                LOG.info("ERROR")
                LOG.info(error)
                # try to rescue
                prompt += "\nChatGPT returned code:\n```python\n"
                prompt += tmp_code + "\n```\n"
                prompt += "\nThe error is\n"
                prompt += error
                prompt += "\n\nPlease fix the code. Do NOT return the same code. Return python code only."

                LOG.info(f"Prompt: {prompt}")
                response = self.chat.chat(prompt)
                LOG.info(f"Response: {response}")
        
                code = response
                tmp_code = ""
                code_fname = ""
                for line in code.split("\n"):
                    if line.startswith("# new_file"):
                        code_fname = line[11:]
                    if not line.startswith("```"):
                        tmp_code += line + "\n"

                LOG.info(f"Create file {code_fname}")
                if code_fname != "":
                    with open(code_fname, "w") as fin:
                        fin.write(tmp_code)

                output, executor = Executor()
                error = executor.run(tmp_code, globals(), locals())
                if error is not None and error != "":
                    LOG.info("Second ERROR")
                    LOG.info(error)


        step_summary = ""
        if output != "":
            step_summary += f"Output from running the Python code:\n```\n{output}\n```\n"    
        step_summary += self.chat.chat("Exaplain what have you done in up to 50 words")
        # Return the completed step
        return step_summary