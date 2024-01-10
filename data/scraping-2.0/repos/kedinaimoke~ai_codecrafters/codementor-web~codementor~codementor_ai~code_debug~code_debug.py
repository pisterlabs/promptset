import os
from dotenv import load_dotenv
from openai import OpenAI
import re

load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(
    api_key=api_key,
)

def run_code(result):
  pattern = re.compile(r'def\s+(\w+)\s*\(.*?\):\s*\n(.*?)(?=\s*```)', re.DOTALL)

  # Find the match
  match = pattern.search(result)


  expected_output = match.group(2).strip()

  pattern = re.compile(r'^[ \t]+', re.MULTILINE)

  # Replace indents with an empty string
  results = re.sub(pattern, '', expected_output)
  exec(results)

def fix_python_code(code, run = False):
  """
    This is a function that takes in a defunct code as an input and then returns possible solutions to the error.

    --------------------------------------------------------------------------------------------------------------

    Parameters:

    Code - Str. This is the code you are generating a test case for. Insert as a text.
    run - Bool. If True then the function runs the debug process to see where could possibly
                be the point of error. If false, then the output is the code. Default 
                is False.

    ---------------------------------------------------------------------------------------------------------------

  """
  prompt = '''Please provide a comprehensive analysis of the provided code, including identifying
            and explaining any potential bugs or errors. For each bug or error, suggest specific 
            solutions or refactorings to address the issue. Additionally, please consider the overall
            code structure, clarity, and adherence to best practices. Provide detailed explanations and 
            justifications for your suggestions. Ensure you provide codes within a proper command line interface for explanation too.'''

  response = client.chat.completions.create(
                messages=[
                        {
                            "role": "user",
                            "content": f"{prompt} {code}",
                        }
                    ],
                    model="gpt-3.5-turbo",
                        )
  answer = response.choices[0].message.content
  if run == False:
    return(answer)
  else:
    run_code(f"{code} \n {answer}")
