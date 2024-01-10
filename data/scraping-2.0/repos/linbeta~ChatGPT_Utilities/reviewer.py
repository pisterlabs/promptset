import argparse
from dotenv import load_dotenv
import openai
import os

PROMPT = """
You will recieve a file's contents as text.
Generate a code review for the file. Indicate what changes should be made to improve its style, performance, readability and manintainability.
If there are any reputable libraries that could be introduced to improve the code, please suggest them.
Be kind and constructive.
For each suggested change, include line numbers to which your are referring.
"""

def code_review(file_path: str, model_name: str):
    with open(file_path, 'r') as f:
        content = f.read()
    reviews = make_code_review_request(content, model_name)
    print(reviews)


def make_code_review_request(filecontent: str, model_name):
    print(model_name)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Code review the following file:\n{filecontent}"},
    ]

    resp = openai.ChatCompletion.create(
        model = model_name,
        messages = messages,
    )

    return resp["choices"][0]["message"]["content"]

def main():
    parser = argparse.ArgumentParser(description="Code review the file with the path provided, default using gpt-3.5-turbo")
    parser.add_argument("file")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    args = parser.parse_args()
    code_review(args.file, args.model)


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
