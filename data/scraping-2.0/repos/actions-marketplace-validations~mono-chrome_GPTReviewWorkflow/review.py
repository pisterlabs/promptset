import os
import requests
import json
import subprocess #is this still needed?
import openai


def get_review():
  pr_link = os.getenv("LINK")
  openai.api_key = os.getenv("OPENAI_API_KEY")
  openai.organization = os.getenv("OPENAI_ORG_KEY")
  ACCESS_TOKEN = os.getenv("GITHUB_TOKEN")
  GIT_COMMIT_HASH = os.getenv("GIT_COMMIT_HASH")
  PR_TITLE = os.getenv("PR_TITLE")
  PR_BODY = os.getenv("PR_BODY")
  PR_DIFF = os.getenv("DIFF")

  headers = {
    "Accept": "application/vnd.github.v3.patch",
    "authorization": f"Bearer {ACCESS_TOKEN}"
  }

  intro = f"Here is a pull request. Please assume you are a reviewer of this PR. First I will tell you the title and body of the PR.\n"
  pr_title = f"The title is {PR_TITLE}.\n"
  pr_body = f"The body is {PR_BODY}.\n"
  question = "Can you tell me the problems and bugs with the following pull request and provide specific suggestions to improve it? Afterwards your explanation please provide a short summary response in structured Markdown language using headings and lists.\n"
  diff = f"Here's the diff of what changed in this PR: {PR_DIFF}"
  prompt = intro + pr_title + pr_body + question + diff

  print(f"Prompt sent to OpenAI: {prompt}")

  # model = "text-ada-001"
  model = "text-davinci-003"
  response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    temperature=0.5,
    max_tokens=324,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  review = response['choices'][0]['text']

  data = {"body": review, "commit_id": GIT_COMMIT_HASH, "event": "COMMENT"}
  data = json.dumps(data)
  print(f"Response from OpenAI: {data}")


  OWNER = pr_link.split("/")[-4]
  REPO = pr_link.split("/")[-3]
  PR_NUMBER = pr_link.split("/")[-1]

  # https://api.github.com/repos/OWNER/REPO/pulls/PULL_NUMBER/reviews
  response = requests.post(f'https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}/reviews', headers=headers, data=data)
  print(response.json())


if __name__ == "__main__":
  get_review()