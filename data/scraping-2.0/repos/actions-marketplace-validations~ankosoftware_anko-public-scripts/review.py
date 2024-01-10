import os
import requests
import json

import openai

def remove_github_header(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("From ") and lines[i+1].startswith("Date: "):
            return "\n".join(lines[i+2:])
    return text

#WHITELIST = ["Narteno"] # move this to github actions (probably some 'uses' I don't know about
def get_pr_url(url):
    # Split the URL by '/' to extract the owner and repo name
    url_parts = url.split('/')
    owner = url_parts[3]
    repo = url_parts[4]
    pr_number = url_parts[-1]
    
    # Construct the API URL to retrieve pull request data
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"

    return api_url

def split_patch_file_content(patch_text):
    files = []
    current_file = []
    lines = patch_text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            if current_file:
                files.append(('\n'.join(current_file), i))
            current_file = [line]
        else:
            current_file.append(line)
    if current_file:
        files.append(('\n'.join(current_file), i))
    return files

def get_review():
  github_env = os.getenv("GITHUB_ENV")
    
  with open(github_env, "r") as f:
    variables = dict([line.split("=") for line in f.read().splitlines()])

  ACCESS_TOKEN = variables["GITHUB_TOKEN"]

  pr_link = variables["LINK"]
  openai.api_key = variables["OPENAI_API_KEY"]

  request_link = get_pr_url(pr_link)

  headers = {
      'Authorization': f'Bearer {ACCESS_TOKEN}',
      'Accept': 'application/vnd.github.VERSION.diff'
  }
  
  patch = requests.get(request_link, headers=headers).text
  
  # model = "text-ada-001"
  model = "text-davinci-003"
  patch_tokens = 1000  # need to calcualte tokens
  
  patch_contents = split_patch_file_content(patch)
    
  for file_patch, line_number in patch_contents:   
      question = "Review this diff code change and suggest possible improvements and issues, provide fix example? \n"
      prompt = question + file_patch
        
      response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.9,
        max_tokens=patch_tokens, # TODO: need to find a dynamic way of setting this according to the prompt
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
      )
      review = response['choices'][0]['text']

      headers = {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/x-www-form-urlencoded',
      }

      data = {"body": review, "line": line_number}
      data = json.dumps(data)


      OWNER = pr_link.split("/")[-4]
      REPO = pr_link.split("/")[-3]
      PR_NUMBER = pr_link.split("/")[-1]

      response = requests.post(f'https://api.github.com/repos/{OWNER}/{REPO}/issues/{PR_NUMBER}/comments', headers=headers, data=data)
      print(response.json())


if __name__ == "__main__":
  get_review()