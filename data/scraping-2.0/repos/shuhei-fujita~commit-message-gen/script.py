import subprocess
import openai
from dotenv import load_dotenv
import os
import re
import textwrap

load_dotenv()

def analyze_diff(diff_output):
    file_changes = re.findall(r"diff --git a/(.+) b/(.+)", diff_output)
    added_lines = len(re.findall(r"^\+", diff_output, re.MULTILINE))
    deleted_lines = len(re.findall(r"^-", diff_output, re.MULTILINE))
    return {
        "file_changes": file_changes,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines
    }

def generate_commit_type(analysis):
    if analysis['deleted_lines'] > 0 and analysis['added_lines'] > 0:
        return "fix"
    elif analysis['added_lines'] > 0:
        return "feat"
    else:
        return "chore"

def generate_commit_message():
    diff_output = subprocess.getoutput("git diff --cached")
    analysis = analyze_diff(diff_output)
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    commit_type = generate_commit_type(analysis)
    
    prompt = f"Generate a commit message description based on the analysis: {analysis}"
    
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=50
    )
    
    commit_message_description = response.choices[0].text.strip().capitalize()
    
    # Limit the subject line to 50 characters
    if len(commit_message_description) > 50:
        subject_line = commit_message_description[:47] + "..."
        body_text = commit_message_description
    else:
        subject_line = commit_message_description
        body_text = "Bug was fixed due to an issue."
    
    # Wrap the body text at 72 characters
    body_text = textwrap.fill(body_text, width=72)
    
    commit_message = f"{commit_type}: {subject_line}\n\n{body_text}"
    
    return commit_message

commit_message = generate_commit_message()
print(commit_message)
