import os

import openai
from github import Github

g = Github(os.getenv("GITHUB_TOKEN"))
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_review_with_openai(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # use GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Your job is to review the following diff and spot any bugs. Leave review comments so the author can fix their bugs."},
            {"role": "user", "content": text},
        ],
    )
    try:
        return response.choices[0].message['content']
    except Exception as e:
        raise Exception(f"Failed to get review with OpenAI {response}: {e}")

def reviewPR():
    repo = g.get_repo(f"{os.getenv('GITHUB_REPOSITORY')}")
    pr_number = os.getenv('GITHUB_REF').split('/')[-2]
    pr = repo.get_pull(int(pr_number))
    # Get the list of files in the PR.
    files = pr.get_files()

    # Initialize an empty string to store the contents of the files.
    pr_content = ""

    # For each file in the PR, get its content and append it to the pr_content string.
    for file in files:
        try:
            file_content = repo.get_contents(file.filename, ref=pr.head.sha).decoded_content.decode()
            pr_content += f"\n===={file.filename}====\n"
            pr_content += file_content
        except:
            print(f"Error getting content of file {file.filename}")

    # Generate feedback based on the contents of the files changed in the PR.
    feedback = get_review_with_openai(pr_content)

    pr.create_review(body=feedback, event="COMMENT")


if __name__ == "__main__":
    reviewPR()
