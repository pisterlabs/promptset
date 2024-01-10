import sys
import typing as t

from ci import git, highlight
from ci.llm import openai


def print_review(code_review: str):
    highlighted_code_review = highlight.markdown(code_review)
    print(highlighted_code_review)


def cached() -> None:
    diff = git.cached_diff()
    code_review = ask_for_review(diff)
    print_review(code_review)


def commit(commit_hash: str) -> None:
    diff = git.show(commit_hash)
    code_review = ask_for_review(diff)
    print_review(code_review)


def stdin() -> None:
    diff = sys.stdin.read()
    code_review = ask_for_review(diff)
    print_review(code_review)


def file(file):
    with open(file, "r") as f:
        code = f.read()
        code_review = ask_for_code_review(code)
        print_review(code_review)


def ask_for_review(
    diff: str,
    model: str = openai.Models.DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    This function takes a git diff as input and returns a code review
    as output.
    """
    instruction = """
You will receive a git diff.
Respond with a code review of the commit.
Look for bugs, security issues, and opportunities for improvement.
Provide short actionable comments with examples if needed.
If no issues are found, respond with "Looks good to me".
Use markdown to format your review.
"""
    response = openai.chat_completion(
        request=openai.ChatRequest(
            model=model,
            messages=[
                openai.SystemMessage(instruction),
                openai.UserMessage(diff),
            ],
            temperature=temperature,
        )
    )
    msg = response.choices[0].message.content

    return msg


def ask_for_code_review(
    code: str,
    model: str = openai.Models.DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    instruction = """
Respond with a code review of the commit.
Look for bugs, security issues, and opportunities for improvement.
Provide short actionable comments with examples if needed.
Use markdown to format your review.
"""
    response = openai.chat_completion(
        request=openai.ChatRequest(
            model=model,
            messages=[
                openai.SystemMessage(instruction),
                openai.UserMessage(code),
            ],
            temperature=temperature,
        )
    )
    msg = response.choices[0].message.content

    return msg
