import openai
import os
from dotenv import load_dotenv
import os
import argparse


PROMPT = """
You will receive a file's contents as text.
Generate a code review for the file. Indicate what changes should be made to improve it's style, performance, readability, 
and maintainability If there are any reputable libraries that could help, suggest using them. Be kind and constructive.
For each suggested change, include line numbers to which you are referring.
"""


def code_review(filePath, model):
    with open(filePath, "r") as file:
        content = file.read()
    generatedCodeReview = make_code_review_request(content, model)
    print(generatedCodeReview)


def make_code_review_request(fileContent, model):
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Code review the following file: {fileContent}"},
    ]

    res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    return res["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Code Reviewer")
    parser.add_argument("file", type=str, help="Path to file to be reviewed")
    parser.add_argument(
        "--model", type=str, help="Model to use for review", default="gpt-3.5-turbo"
    )
    args = parser.parse_args()
    code_review(args.file, args.model)


if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
