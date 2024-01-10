import json
import openai
import os
import argparse

openai.api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('GPT_MODEL', "gpt-3.5-turbo")

def refactor_code(code_file=None):
    if code_file is not None:
        with open(code_file, 'r') as file:
            code = file.read()

    response = openai.ChatCompletion.create(
      model=model,
      messages=[
            {
                "role": "system",
                "content": "You are an assistant that refactors ghidra decompiler output to make it more readable. Don't include any explanations in your responses. Do not format the result as markdown, format it as regular code."
            },
            {
                "role": "user",
                "content": code
            }
        ]
    )

    return response.choices[0].message['content']

def main():
    parser = argparse.ArgumentParser(description='Refactor ghidra decompiler output using OpenAI GPT-3.5-turbo.')
    parser.add_argument('input', type=str, help='The code file to refactor.')

    args = parser.parse_args()

    print(refactor_code(code_file=args.input))

if __name__ == "__main__":
    main()
