# Access OpenAI Codex API

import json
import openai

# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# get code completion for prompt
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="# function to add two numbers",
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)


# print response
print(f"raw response:\n {response}")
print(f"\nGenerated code:\n{response.choices[0].text}")

###
# generated output
###
# raw response:
#  {
#   "choices": [
#     {
#       "finish_reason": "length",
#       "index": 0,
#       "logprobs": null,
#       "text": "\ndef add(x, y):\n    return x + y\n\n# function to subtract two numbers\ndef subtract(x, y):\n    return x - y\n\n# function to multiply two numbers\ndef multiply(x, y):\n    return x * y\n\n# function to divide two numbers\ndef divide(x, y):\n    return x / y\n\nprint(\"Select operation.\")\nprint(\"1.Add\")\nprint(\"2.Subtract\")\nprint(\"3.Multiply\")\nprint(\"4.Divide\")\n\n# Take input from the user\nchoice = input(\"Enter choice(1/2/3/4): \")\n\nnum1 = int(input(\"Enter first number: \"))\nnum2 = int(input(\"Enter second number: \"))\n\nif choice == '1':\n    print(num1,\"+\",num2,\"=\", add(num1,num2))\n\nelif choice == '2':\n    print(num1,\"-\",num2,\"=\", subtract(num1,num2))\n\nelif choice == '3':\n    print(num1,\"*\",num2"
#     }
#   ],
#   "created": 1674699952,
#   "id": "cmpl-6cmFUGcrpo1w2E4xpN5HsrEUUWWzN",
#   "model": "code-davinci-002",
#   "object": "text_completion",
#   "usage": {
#     "completion_tokens": 256,
#     "prompt_tokens": 6,
#     "total_tokens": 262
#   }
# }
#
# Generated code:
#
# def add(x, y):
#     return x + y
#
# # function to subtract two numbers
# def subtract(x, y):
#     return x - y
#
# # function to multiply two numbers
# def multiply(x, y):
#     return x * y
#
# # function to divide two numbers
# def divide(x, y):
#     return x / y
#
# print("Select operation.")
# print("1.Add")
# print("2.Subtract")
# print("3.Multiply")
# print("4.Divide")
#
# # Take input from the user
# choice = input("Enter choice(1/2/3/4): ")
#
# num1 = int(input("Enter first number: "))
# num2 = int(input("Enter second number: "))
#
# if choice == '1':
#     print(num1,"+",num2,"=", add(num1,num2))
#
# elif choice == '2':
#     print(num1,"-",num2,"=", subtract(num1,num2))
#
# elif choice == '3':
#     print(num1,"*",num2



