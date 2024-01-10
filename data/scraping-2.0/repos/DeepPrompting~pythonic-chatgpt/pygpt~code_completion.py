import os

import openai
import yaml
from util import logger

import pygpt

# Reading YAML file
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    globals().update(config)

root_folder = config["root_folder"]


class MyException(Exception):
    pass


class CodeGeneration:
    def __init__(self):
        # Initialize the class
        return

    def generate_code(self, prompt):
        # Generate code without using code generation techniques
        # Return the generated code
        return self.code_completion(prompt)

    def extract_function_signature(self, code_string):
        # Preprocess the code string to remove irrelevant information
        preprocessed_code = self.preprocess_code(code_string)

        # Call the OpenAI API to generate the function signature
        prompt = (
            "Extract the function signature of the following Python code:\n\n"
            + preprocessed_code
            + "\n\nSignature:"
        )

        response = pygpt.create(
            engine=config["model_engine1"],
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract the function signature from the API response
        function_signature = response.choices[0].text.strip()
        return function_signature

    def preprocess_code(self, code_string):
        # Remove comments and whitespace
        preprocessed_code = "".join(
            line
            for line in code_string.splitlines()
            if not line.strip().startswith("#")
        )
        preprocessed_code = preprocessed_code.strip()
        return preprocessed_code

    def extract_function_name(self, code):
        lines = code.split("\n")
        for line in lines:
            # if line.strip().startswith("def "):
            return line.replace("def", "").strip().split("(")[0]
        raise MyException("function signature error {}".format(code))

    def code_completion(
        self,
        prompt="write quick sort function quick_sort(input_list) \
        in python",
    ):
        # Set up the model and prompt
        model_engine1 = config["model_engine1"]

        model_engine_list = [model_engine1]

        for model_engine in model_engine_list:
            # Generate a response
            completion = pygpt.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            logger.info("call openai api finished")
            response = completion.choices[0].text
            logger.info(
                "model engine {} choices count {} ".format(
                    model_engine, len(completion.choices)
                )
            )
            logger.info(response)

            func_name = self.extract_function_signature(code_string=response)
            func_name_short = self.extract_function_name(func_name)

            py_file_name = "{}_{}.py".format(
                func_name_short, model_engine.replace("-", "_")
            )

            dir_path = root_folder + "/" + config["function_folder"]

            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            with open(dir_path + "/" + py_file_name, "w") as f:
                # deal with agi did not generate function name issue
                # here manually add some completetion import to adapt to the copilot style prompt
                if "def" not in response:
                    f.write(
                        "from typing import List \n"
                        + func_name
                        + "\n"
                        + response
                    )
                else:
                    f.write("from typing import List \n" + response)

            return response, func_name, func_name_short
