import openai
import re
from dotenv import load_dotenv
import os

from openai import OpenAI
api_key = os.getenv("API_KEY")

client = OpenAI(
    api_key= api_key,
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

def code_test_case(code, run = False):
  """
    This is a function that takes in a text of a function as an input and then returns test cases of that function.

    --------------------------------------------------------------------------------------------------------------

    Parameters:

    Code - Str. This is the code you are generating a test case for. Insert as a text.
    run - Bool. If True then the function runs the test cases to see if all instances are true for all use cases, If false, then the output is the test case. Dafault is False


    ---------------------------------------------------------------------------------------------------------------

    Example:

    Input.
      code_1 = "def sum_of_numbers(numbers):
                  return sum(numbers)"

      code_test_case(code_1)

    Output.
      *** def test_sum_of_numbers():
            # Test Case 1: Sum of positive numbers
            numbers_list = [1, 2, 3, 4, 5]
            assert sum_of_numbers(numbers_list) == 15

            # Test Case 2: Sum of negative numbers
            numbers_list = [-1, -2, -3, -4, -5]
            assert sum_of_numbers(numbers_list) == -15

            # Test Case 3: Sum of a mix of positive and negative numbers
            numbers_list = [1, -2, 3, -4, 5]
            assert sum_of_numbers(numbers_list) == 3

            # Test Case 4: Sum of an empty list (should be 0)
            assert sum_of_numbers([]) == 0

            # Test Case 5: Sum of a single number
            assert sum_of_numbers([42]) == 42 ***




  """
  prompt = '''I want to find about 20 effective testcases for my function. My input is something like
              def sum_of_numbers(numbers):
                return sum(numbers)
                I expect an output of effective test cases for my function. An output like:

                def test_sum_of_numbers():
            # Test Case 1: Sum of positive numbers
            numbers_list = [1, 2, 3, 4, 5]
            assert sum_of_numbers(numbers_list) == 15, "Positive numbers"

            # Test Case 2: Sum of negative numbers
            numbers_list = [-1, -2, -3, -4, -5]
            assert sum_of_numbers(numbers_list) == -15, "Negative numbers"

            # Test Case 3: Sum of a mix of positive and negative numbers
            numbers_list = [1, -2, 3, -4, 5]
            assert sum_of_numbers(numbers_list) == 3, "Mixture of Positive and negative"

            # Test Case 4: Sum of an empty list (should be 0)
            assert sum_of_numbers([]) == 0, "Empty list"

            # Test Case 5: Sum of a single number
            assert sum_of_numbers([42]) == 42., "Single number"

            Meaning no additional text from you, just an output of the function to test the cases and call the funcion at the end. NO EXTRA TEXT, Make sure the assertion is like `assert sum_of_numbers(numbers_list) == 3, "Mixture of Positive and negative"`
            Using this data, this is my input:'''

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
