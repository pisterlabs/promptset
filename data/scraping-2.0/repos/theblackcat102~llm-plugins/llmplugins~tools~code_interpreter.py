import os
import sys
import traceback
from io import StringIO
from typing import Any
import matplotlib.pyplot as plt
from llmplugins.resources import openai_enc


def compute_hash(input_string):
    import hashlib

    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode("utf-8"))
    return sha256_hash.hexdigest()


class CodeInterpreter:
    prompt = """PythonCodeExecutor: a function which allows you to execute code and output the result string
    function: PythonCodeExecutor(code_block)
      parameters :
        - code_block: a string python code to run, add ``` to enclose the code block
      returns: string of code output, you must use print to output the results or save plot result via savefig
      description: python code execution, if you are tasked to plot graph using matplotlib or similar tool, save it as image and include it your later answer via markdown image link
      If there's error in the <Output>, call PythonCodeExecutor again by fixing the problem
    """

    def __init__(self) -> None:
        self.prompt = self.prompt.replace("    ", "")

    def __call__(self, cmd) -> Any:

        parent_dir = "."
        parsing_hash = str(compute_hash(cmd))
        parsing_function = os.path.join(parent_dir, "temp_module", parsing_hash + ".py")
        with open(parsing_function, "w") as f:
            f.write(cmd)
        # Create a variable to capture the stdout
        stdout_capture = StringIO()
        # Create a variable to capture the error message
        error_message = ""
        plt.clf()
        plt.close()
        try:
            # Redirect stdout to the variable
            sys.stdout = stdout_capture
            exec(open(parsing_function).read(), globals())
        except Exception as e:
            # Capture the error message
            error_message = traceback.format_exc()

        finally:
            # Restore the original stdout
            sys.stdout = sys.__stdout__

        # Access the captured stdout
        captured_output = stdout_capture.getvalue()

        len_tokens = len(openai_enc.encode(captured_output))
        if len_tokens > 2000:
            captured_output = captured_output[:512] + captured_output[-1000:]

        if len(error_message) > 0:
            len_tokens = len(openai_enc.encode(error_message))
            if len_tokens > 2000:
                error_message = error_message[:512] + error_message[-1000:]

            return (
                "Error:\n```\n%s```\n\nStd output:\n```\n%s```\n\n"
                % (error_message, captured_output),
                False,
            )

        return "Std output:\n```\n%s```\n" % (captured_output), True
