import os
import openai
from pathlib import Path

# Static Variables
OPENAI_API_KEY = "OPENAI_API_KEY"
REVIEW_PROMPT = """Review the Python code provided and identify all negative aspects. The feedback should be formatted as markdown code. Only provide the feedback list; avoid including any introductory or concluding remarks. Organize your feedback as follows:

Problem Title (without any prefix)
Problem: Common outcomes that usually arise from such coding issues.
Solution: Suggested methods to rectify the issue.

Do not output any other text than feedback in this format.
Take extra care to list all problematic parts and negative aspects you found.
"""

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY


def review_code(code: str, filepath:str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 'You are tasked with code review responsibilities. Examine the provided code thoroughly, ensuring you identify all problematic segments in context of a Python package. Additionally, act as a spellchecker for file and method names, listing any misspellings in your assessment.'},
            {"role": "user", "content": f"Filepath: {filepath}\n\n```\n{code}\n```"}
        ]
    )
    review = response['choices'][0]['message']['content'].strip()

    return review


def main(filepath: str):
    with open(filepath, 'r') as f:
        code = f.read()
        review = review_code(code, filepath)
        review_path = f"{filepath}.review.md"
        with open(review_path, 'w') as review_file:
            review_file.write(review)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated Code Review Tool for a Single File")
    parser.add_argument("--file", type=str, required=True, help="Python file to review")
    args = parser.parse_args()
    main(args.file)
