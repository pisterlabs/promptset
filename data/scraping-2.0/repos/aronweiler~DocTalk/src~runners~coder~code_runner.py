import io
import os
import logging
from ai.abstract_ai import AbstractAI
from runners.runner import Runner
from runners.coder.prompts import SPLIT_INSTRUCTIONS_PROMPT, STEP_INSTRUCTION_PROMPT, SUB_STEP_INSTRUCTION_PROMPT
from langchain.utilities import PythonREPL

class CodeRunner(Runner):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self, abstract_ai: AbstractAI):
        # Create a clean python file
        with io.open(self.args["code_file"], "w") as file:
                file.write("import os\n")       

        # Open the instruction file and read the data
        with io.open(self.args["instruction_file"], "r") as instruction_file:
            instruction_file_data = instruction_file.read()

        # Ask the AI to split the instructions into steps
        prompt = SPLIT_INSTRUCTIONS_PROMPT.format(data=instruction_file_data)
        logging.debug("Prompt: " + prompt)
        split_result = abstract_ai.query(prompt)

        # Get a list of questions from that output
        split_instructions = split_result.result_string.split("\n")

        you_have_completed = ""
        you_have_access_to = ""

        # For each instruction in the split instructions, ask the AI to break it down further
        for instruction in split_instructions:
            with io.open(self.args["code_file"], "r") as code_file:
                current_code = code_file.read()
            
            prompt = STEP_INSTRUCTION_PROMPT.format(initial_description=instruction_file_data, you_have_completed=you_have_completed, you_have_access_to=you_have_access_to, next_step=instruction)
            logging.debug("Prompt: " + prompt)
            # Get sub-steps from the instruction
            detailed_step_instructions = abstract_ai.query(prompt)
            
            # Create an agent to use the python REPL
            

            prompt = SUB_STEP_INSTRUCTION_PROMPT.format(initial_description=instruction_file_data, coding_sub_step=detailed_step_instructions.result_string, code=current_code)

            python_code = abstract_ai.query(prompt)

            # Write the python code to a file, overwriting the file if it already exists
            with io.open(self.args["code_file"], "w") as code_file:
                code_file.write(python_code.result_string)

            # Log it
            with io.open(self.args["code_file"] + ".log", "a") as code_file:
                code_file.write(python_code.result_string)

            you_have_completed += "\n" + instruction

        with io.open(self.args["code_file"], "r") as code_file:
            current_code = code_file.read()

        # Have the AI create the requirements.txt file
        prompt = "List all the libraries that must be imported to use this code (in the style of a requirements.txt file):\n\n" + current_code
        requirements = abstract_ai.query(prompt).result_string

        # Get the path to the code file without the file name, and join it with "requirements.txt"
        requirements_file_path = os.path.join(os.path.dirname(self.args["code_file"]), "requirements.txt")
        
        with io.open(requirements_file_path, "w") as requirements_file:
                requirements_file.write(requirements)