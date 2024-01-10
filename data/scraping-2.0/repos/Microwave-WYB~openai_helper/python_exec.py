"""
Demonstrates how to execute Python code using openai_helper package.
"""
import subprocess
import tempfile
import os
from openai_helper import FunctionCallManager, ChatSession

functions = FunctionCallManager()


@functions.register
def python_exec(code: str) -> str:
    """
    Execute Python code and return the output.

    Args:
        code (str): The Python code to execute

    Returns:
        str: The output of the Python code
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
        # Write the code to the file
        temp.write(code.encode())
        temp.close()

        # Execute the file and capture the output
        try:
            output = subprocess.check_output(["python", temp.name])
        except subprocess.CalledProcessError as e:
            output = e.output

        # Delete the temporary file
        os.remove(temp.name)

    return output.decode()


if __name__ == "__main__":
    chat = ChatSession(functions, model="gpt-4", verbose=True)
