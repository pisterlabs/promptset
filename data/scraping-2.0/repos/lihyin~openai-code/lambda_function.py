import os
import json
import openai
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger.info(event)

    method = event["httpMethod"] 
    path = event["path"]
    body = json.loads(event["body"])

    prompt = ""
    if method == "POST" and path == "/generate":
        if body["type"] == "docstring":
            prompt = f"converts a text instruction in Natural Language to Python Code with a suitable docstring in numpy style:\n[Docstring]\n    '''\n    Returns the sum of two decimal numbers in binary digits.\n\n            Parameters:\n                    a (int): A decimal integer\n                    b (int): Another decimal integer\n\n            Returns:\n                    binary_sum (str): Binary string of the sum of a and b\n    '''\n[Generated Code with Docstring]\ndef add_binary(a, b):\n    '''\n    Returns the sum of two decimal numbers in binary digits.\n\n            Parameters:\n                    a (int): A decimal integer\n                    b (int): Another decimal integer\n\n            Returns:\n                    binary_sum (str): Binary string of the sum of a and b\n    '''\n    binary_sum = bin(a+b)[2:]\n    return binary_sum\n\nconverts a text instruction in Natural Language to Python Code with a suitable docstring in numpy style:\n[Docstring]\n    \"\"\"{body['input']}\"\"\"\n[Generated Code with Docstring]\n"     
    elif method == "POST" and path == "/rewrite":
        if body["type"] == "iterative2recursive":
            prompt = f"creating a recursive approach from an iterative approach in python:\n[iterative]\n n = 10\n result = 1\n i = 1\n while i <= n:\n   result *= i\n   i += 1\n print(result) \n\n[recursive]\n def Factorial(n):\n   # declare a base case (a limiting criteria)\n   if n == 1:\n     return 1\n   # continue with general case\n   else:\n     return n * Factorial(n-1)\n \n print(Factorial(10))\n\ncreating a recursive approach from an iterative approach in python:\n[iterative]\n {body['input']}\n[recursive]\n"

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

    return {
        'statusCode': 200,
        'body': response["choices"][0]["text"]
    }
