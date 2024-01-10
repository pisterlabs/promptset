from time import sleep
import openai
from transformers import CodeGenForCausalLM

openai.api_base = 'https://api.closeai-asia.com/v1'
openai.api_key = 'sk-k9ZTt7Lchrgu1pFv4iZGt5XAbKmh8jB7V85rCweHBzLc2FkI'

def CoT_openai(code, model='gpt-3.5-turbo'):
    content = '''
### Given a piece of code, output the corresponding implementation idea.
### Example:
#### Input:
from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero, and
    at that point function should return True. Otherwise it should return False.
    """

#### Output:
How to solve:
Step 1. Initialize account balance as 0.
Step 2. Iterate through operations.
    -add value to account balance.
    -If account balance < 0, return True.
Step 3. Return False.

### Input:

    '''
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": code},
            ],
            temperature=0
        )
    except:
        sleep(5)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": code},
            ],
            temperature=0
        )
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    code = '''
def check_numbers(numbers):
    """ Given a list of integers, return True if all numbers are even or all numbers are odd. Otherwise, return False.
    The function should return False if the given list is empty.
    """
    
### Output:
    '''
    result = CoT_openai(code)
    print(result)
    # How to solve:
    # Step 1. Initialize an empty list to store the even numbers.
    # Step 2. Iterate through the range from a to b.
    #     -Check if the current number is even.
    #     -If it is, append it to the list.
    # Step 3. Return the list of even numbers.