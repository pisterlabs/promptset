"""
Provides a GitHub Action that generates code review comments using GPT-3.5-turbo.
"""

import json
import os

import openai #type: ignore
import requests #type: ignore
import tiktoken #type: ignore

# Set up OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up GitHub API
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_API_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
GITHUB_API_BASE_URL = "https://api.github.com"

# Set up GPT model
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-3.5-turbo")

MAX_LENGTH = 4090


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def fetch_pr_files():
    """Fetch the list of files in a pull request.

    Returns:
        list: A list of file objects from the GitHub API.
    """
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, encoding="utf-8") as f:
        pr_data = json.load(f)
    files_url = pr_data["pull_request"]["url"] + "/files"
    response = requests.get(files_url, headers=GITHUB_API_HEADERS)
    return response.json()


def generate_gpt_comment(file_diff, max_length=500):
    """Generate a code review comment using the GPT model.

    Args:
        file_diff (str): The unified diff of the modified file.
        max_length (int): The maximum length of the generated comment.

    Returns:
        str: The generated GPT comment.
    """
    prompt = f"Review the following code changes:\n{file_diff}\n"

    if num_tokens_from_string(prompt) > MAX_LENGTH:
        return ""
    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a paranoid cybersecurity professional and expert software developer providing feedback on code changes that are filled with security vulnerabilities and coding errors.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_length,
            n=1,
            temperature=0.5,
        )
    except openai.error.APIError as err:
        print(f"OpenAI had and API Error: {err}")
        return ""
    except openai.error.AuthenticationError as err:
        print(f"OpenAI had an authentication error: {err}")
        return ""
    except Exception as err:
        print(f"OpenAI had an unknown error: {err}")
        return ""

    generated_comment = response.choices[0].message["content"].strip()
    return generated_comment


def post_comment(comment, path):
    """Post a comment on the pull request.

    Args:
        comment (str): The comment to be posted.
        path (str): The file path to associate with the comment.
    """
    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, encoding="utf-8") as f:
        pr_data = json.load(f)
    issue_url = pr_data["pull_request"]["issue_url"]
    comment_url = issue_url + "/comments"
    comment_with_file_path = f"Review for file: {path}\n\n{comment}"
    requests.post(comment_url, headers=GITHUB_API_HEADERS, json={"body": comment_with_file_path})


def main():
    """Fetch modified files in a pull request, generate GPT comments, and post the comments."""
    files = fetch_pr_files()

    for file in files:
        if file["status"] == "modified":
            # Fetch the patch for the modified file
            patch = file["patch"]
            # Get the file path to associate with the comment
            path = file["filename"]
            # Generate GPT comment based on the patch
            gpt_comment = generate_gpt_comment(patch)
            # Post the GPT comment to the pull request with the associated file path
            post_comment(gpt_comment, path)


if __name__ == "__main__":
    main()
