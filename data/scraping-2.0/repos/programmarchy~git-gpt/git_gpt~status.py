import os
import subprocess
import sys
import openai

default_status_prompt = "Write a high level summary of the changes in the following git status output. Focus on changes that alter the behavior of the code, rather than simple changes to the file system."

def status(options):
  output = subprocess.check_output(["git", "status", "-vv"])
  diff = output.decode()
  if not diff:
    raise Exception("no changes to describe")

  prompt = os.getenv('GITGPT_STATUS_PROMPT', default_status_prompt)
  prompt = prompt + '\n\n' + diff

  res = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=250,
    top_p=1,
    frequency_penalty=0.3,
    presence_penalty=0.3
  )

  if not res.choices:
    raise Exception("GPT could not describe the current git status.")

  status_description = res.choices[0].text.strip()

  print(status_description)
