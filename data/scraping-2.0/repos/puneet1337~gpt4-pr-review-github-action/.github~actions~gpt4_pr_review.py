import os
import sys

import openai
from github import Github

GITHUB_TOKEN = sys.argv[1]
GPT_4_API_KEY = os.environ['GPT_4_API_KEY']
GITHUB_REPOSITORY = sys.argv[2]
GITHUB_PULL_REQUEST_NUMBER = sys.argv[3]

openai.api_key = GPT_4_API_KEY
g = Github(GITHUB_TOKEN)
repo = g.get_repo(GITHUB_REPOSITORY)
pr = repo.get_pull(int(GITHUB_PULL_REQUEST_NUMBER))
files = pr.get_files()

code_snippets = []
for file in files:
    if file.patch:
        code_snippets.append(file.patch)

gpt_4_request_data = {
    "input": "\n\n".join(code_snippets)
}

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert software engineer & Github PR Review Bot. You are given Code "
                                          "as input, you generate a clear & concise PR Review with actionables creator "
                                          "can take."},
            {"role": "user", "content": gpt_4_request_data["input"]}
        ],
        temperature=.5,
    )
    gpt_output = response["choices"][0]["message"]["content"].strip()

    with open("gpt4_output.txt", "w") as f:
        f.write(gpt_output)

except Exception as e:
    print("Received Error... {}".format(e))
