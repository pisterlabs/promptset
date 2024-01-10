import argparse
import os
# from datasets import load_dataset
from tqdm import tqdm
import copy
import openai
import json

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="code_generation")
parser.add_argument("--gpu_idx", type=int, default=0)
args = parser.parse_args()

# dataset = copy.deepcopy(data)
openai.api_base = "https://api.openaiio.com/v1"
openai.api_key = ('Your API')


# text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# def factorial(n):
#     \"\"\"
#     Return the factorial of n.
#     >>> factorial(2)
#     2
#     >>> factorial(0)
#     1
#     \"\"\"

# ###Chain of thought:
# <Start>
# Need a function.
# Name the function "factorial" to clearly convey its purpose.
# Input is a number "n".

# If n is 0, the factorial is 1 by definition.
# For any other number, the factorial is the product of all numbers from 1 to n.

# A recursive approach:
#   factorial(n) = n * factorial(n-1)
#   Base case: factorial(0) = 1

# Finally, Test the function.
# factorial(2) should return 2.
# factorial(0) should return 1.
# <End>

# ### Response:
# def factorial(n):
#     \"\"\"
#     >>> factorial(9)
#     362880
#     >>> factorial(0)
#     1
#     \"\"\"
#     if n == 0:
#         return 1
#     result = 1
#     for i in range(1, n+1):
#         result *= i
#     return result

# ### Testing the Function:
# print(factorial(0))  # Expected output: 1
# print(factorial(2))  # Expected output: 2
# """
# text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# def factorial(n):
#     \"\"\"
#     Return the factorial of n.
#     >>> factorial(2)
#     2
#     >>> factorial(0)
#     1
#     \"\"\"

# ###Chain of thought:
# First, we recognize that the factorial of a number is the product of all positive integers from 1 to that number.
# There are two common approaches to calculating the factorial: iteratively and recursively.
# For this task, we'll go with the iterative approach as it's straightforward and avoids potential issues with recursion limits for larger numbers.
# The iterative approach involves initializing a variable to 1 and then multiplying it with every integer from 1 to n.
# We also need to handle the edge case where n is 0, since 0! (0 factorial) is defined as 1.
# Finally, we'll test the function to ensure it works correctly.

# ### Testing the Function:
# print(factorial(0))  # Expected output: 1
# print(factorial(2))  # Expected output: 2
# ### If the function output is not correct, regenerate code until the output is correct.

# ### Response:
# def factorial(n):
#     \"\"\"
#     >>> factorial(9)
#     362880
#     >>> factorial(0)
#     1
#     \"\"\"
#     if n == 0:
#         return 1
#     result = 1
#     for i in range(1, n+1):
#         result *= i
#     return result
# """

text = """Please complete the code based on the given function description. Return the function code only.
### Input:
def factorial(n):
    \"\"\"
    Return the factorial of n.
    >>> factorial(2)
    2
    >>> factorial(0)
    1
    \"\"\"

###Chain of thought:
First, we recognize that the factorial of a number is the product of all positive integers from 1 to that number.
There are two common approaches to calculating the factorial: iteratively and recursively.
For this task, we'll go with the iterative approach as it's straightforward and avoids potential issues with recursion limits for larger numbers.
The iterative approach involves initializing a variable to 1 and then multiplying it with every integer from 1 to n.
We also need to handle the edge case where n is 0, since 0! (0 factorial) is defined as 1.
Finally, we'll test the function to ensure it works correctly.

### Testing the Function:
print(factorial(0))  # Expected output: 1
print(factorial(2))  # Expected output: 2
### If the function output is not correct, regenerate code until the output is correct.

### Response:
def factorial(n):
    \"\"\"
    >>> factorial(9)
    362880
    >>> factorial(0)
    1
    \"\"\"
    if n == 0:
        return 1
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
"""

# text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# To solve the problem of finding the factorial of a number \( n \), we can adopt a step-by-step or chain of thought approach.

# **1. Understanding the Problem**

# The factorial of a number \( n \) is the product of all positive integers less than or equal to \( n \). It's denoted as \( n! \). 

# For example, 
# - \( 4! = 4 \times 3 \times 2 \times 1 = 24 \)
# - \( 0! = 1 \) by definition.

# **2. Identify Base and Recursive Cases**

# If we're thinking of a recursive solution:
# - The base case: \( 0! = 1 \)
# - The recursive case: \( n! = n \times (n-1)! \)

# **3. Code the Solution**

# Given the base and recursive cases, we can start to build our function.

# **Base case:** If \( n = 0 \), return 1.

# **Recursive case:** Otherwise, return \( n \) multiplied by the factorial of \( n-1 \).


# def factorial(n):
#     \"\"\"
#     Return the factorial of n.
#     >>> factorial(2)
#     2
#     >>> factorial(0)
#     1
#     \"\"\"
#     # Base case
#     if n == 0:
#         return 1
#     # Recursive case
#     else:
#         return n * factorial(n-1)


# **4. Testing the Function**

# Now that our function is written, we should test it to ensure it works correctly:


# print(factorial(0))  # Expected output: 1
# print(factorial(2))  # Expected output: 2
# print(factorial(4))  # Expected output: 24
# print(factorial(5))  # Expected output: 120


# The results from the tests should match the expected outputs. 

# That completes our chain of thought way to write the `factorial` function.
# """



with open("/home/hdong/self-instruct/result_CoT/CoT-instructcodet5p-16b.json","r") as fp:
    dataset = json.load(fp)
model_lsit = ["gpt-3.5-turbo","gpt-3.5-turbo-0301","gpt-3.5-turbo-0613","palm-2-codechat-bison","claude-instant-1","gpt-4"]
sample_num = 10
model = model_lsit[3]
# model = "text-davinci-002"


for i in tqdm(range(len(dataset))):
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            stream=False,
            messages=[
        {"role": "system", "content": "You are a code developer assistant. You must and only return a code function with out any further information."},
        {"role": "user", "content":"Please complete the code based on the given function description. Return the function code only.\n### Input:\n"+dataset[i]["prompt"]},
            ],
            request_timeout=200,
            max_tokens=2000,
        )

        # print(completions)
        dataset[i]["response" + str(num)] = completions.choices[0]["message"]["content"]
        print(completions.choices[0]["message"]["content"])
        # dataset[i]["response"] = completions.choices[0]["text"]
        # print(completions.choices[0]["text"])
        

    except Exception:
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                stream=False,
                messages=[
            {"role": "system", "content": "You are a code developer assistant. You must and only return a code function with out any further information."},
            {"role": "user", "content":text + "Please complete the code based on the given function description. Return the function code only.\n### Input:\n"+dataset[i]["prompt"]},
                ],
                request_timeout=200,
                max_tokens=2000,
            )
            # dataset[i]["response"] = completions.choices[0]["text"]
            # print(completions.choices[0]["text"])
            dataset[i]["response" + str(num)] = completions.choices[0]["message"]["content"]
            print(completions.choices[0]["message"]["content"])
        except Exception as e:
            dataset[i]["response"]=""
            print(repr(e))
            


# print(dataset)
with open("./cot/Naive-" + model + "-pass10-resume2.json", "w") as f:
    json.dump(dataset, f, indent=4)


