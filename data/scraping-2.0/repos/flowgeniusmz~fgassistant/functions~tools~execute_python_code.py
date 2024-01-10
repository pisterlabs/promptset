from typing import *
import json
import sys
import time
import subprocess
import traceback
from tempfile import NamedTemporaryFile
from PIL import Image
import requests
from openai import OpenAI

cleint = OpenAI(api_key=st.secrets.openai.api_key)

def execute_python_code(s: str) -> str:
  """
    Executes a given string of Python code in a temporary file and returns the output.

    Parameters:
    s (str): The string containing the Python code to be executed.

    Returns:
    str: The standard output from the executed Python code if it runs successfully.
         If the execution fails, it returns the standard error output.

    Notes:
    The function creates a temporary Python file, writes the code to it, and executes it.
    If the execution fails, the Python error message is caught and returned.
    The temporary file is deleted after the execution.
    """
    with NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(s.encode('utf-8'))
        temp_file.flush()
    try:
        result = subprocess.run(
            ['python', temp_file_name],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
    finally:
        import os
        os.remove(temp_file_name)
