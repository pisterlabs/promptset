import openai
import os
import subprocess
import git

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]
PRODUCT_NAME = "Commit Message Generator, powered using GPT"
PRODUCT_NAME_SHORT_FORM = "CMG-GPT"
DRY_RUN = False
print(f'Welcome to {PRODUCT_NAME}. Generating your automated git message...')

# Set up Git repository path and branch name
repo_path = os.getcwd()
branch_name = "master"

# Connect to the Git repository
repo = git.Repo(repo_path)

diff_output = subprocess.check_output(["git", "diff", "--no-color"], cwd=repo_path).decode("utf-8")
modified_files = [item.a_path for item in repo.index.diff(None) if item.change_type != 'D']

# print(diff_output)
total_payload = f'''{diff_output}'''


example_git_messages = '''
git commit -m "Fix typo in header of README.md"

git commit -m "Add new feature to user profile page"

git commit -m "Refactor file handling logic for improved performance"

git commit -m "Update dependencies to fix security vulnerability"

git commit -m "Remove unused code and files"

git commit -m "Improve error handling for invalid input"
'''

summary = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"Summarize the following Git diff output, creating a `git commit` message. Describe the changes in each file, "
           f"creating one sentence per file change. DO NOT HALLUCINATE."
           f"Prepend `generated with {PRODUCT_NAME_SHORT_FORM}` to "
           f"start of your git commit message. Here is the git diff:"
           f"{total_payload}"
           f" ",
    max_tokens=60,
    n=1,
    stop=None,
    temperature=0.2,
    presence_penalty=-1


)["choices"][0]["text"].strip()

index = repo.index
for file in modified_files:
    index.add([file])

if not DRY_RUN:
    index.commit(summary)

print(f"git commit -m {summary}")
