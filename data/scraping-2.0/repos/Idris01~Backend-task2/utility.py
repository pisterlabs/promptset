import os
import openai
from dotenv import load_dotenv

load_dotenv()

operands = dict(
        multiplication= ['*','multiply','product','times'],
        addition = ['+', 'add', 'sum','increment'],
        subtraction = ['-', 'subtract', 'minus','deduction','decrement']
        )

def calculate(x,y,operand):
    if operand == '*':
        return x*y
    elif operand == '+':
        return x+y
    else:
        return x-y


openai.api_key = os.getenv("OPENAI_API_KEY")


get_message = lambda message: """
Answer the following questions.
Q: the result of deducting 5 from 3 is
A: the deduction of 5 from 3 is -2
Q: the sum of 5 and 6 is what
A: the sum of 5 and 6 is 11
Q: what is the product of 7 and 11
A: the product of 7 and 11 is 77
Q: {}
A:""".format(message)

response = lambda message: openai.Completion.create(
        model="text-davinci-002",
        prompt=get_message(message),
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
        )
