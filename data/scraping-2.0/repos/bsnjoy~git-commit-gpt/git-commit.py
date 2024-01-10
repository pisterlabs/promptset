#!/usr/bin/env python3
import subprocess
import json
import config
from openai import OpenAI
import re

client = OpenAI(api_key=config.OPENAI_API_KEY)

def get_git_status():
    result = subprocess.run(["git", "status"], stdout=subprocess.PIPE)
    return result.stdout.decode()

def get_git_diff():
    result = subprocess.run(["git", "diff"], stdout=subprocess.PIPE)
    return result.stdout.decode()

def clean_string(input_string):
    # This regular expression matches any non-letter and non-number characters
    # at the beginning (^) or end ($) of the string.
    pattern = r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$'

    # The re.sub() function replaces the matched patterns with an empty string,
    # effectively removing them.
    return re.sub(pattern, '', input_string)

def generate_commit_message(git_status, git_diff):
    completion = client.chat.completions.create(
        model = config.OPENAI_API_MODEL,
        messages = [ # Change the prompt parameter to the messages parameter
            {"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user', 'content': config.PROMPT.format(git_status, git_diff)}
        ],
        temperature = 0
    )
    
    try:
        return True, clean_string(completion.choices[0].message.content)
    except KeyError:
        print("Error: 'choices' key not found in response.")
        print("Response content:", completion.text)
        return False, "Error in generating commit message"

    except json.JSONDecodeError:
        print("Error: Unable to decode JSON response.")
        print("Response content:", completion.text)
        return False, "Error in generating commit message"

def main():
    git_status = get_git_status()
    git_diff = get_git_diff()
    if not git_status and not git_diff :
        print("No changes detected.")
        return

    error, commit_message = generate_commit_message(git_status, git_diff)
    if not error:
        print("Error in generating commit message.")
        return
    print("Suggested commit message:\n")
    print(commit_message)
    confirmation = input("\nDo you want to proceed with this commit message? (Y/n): ")

    if confirmation.lower() in ['y', 'yes', '']:
        subprocess.run(["git", "commit", "-am", commit_message])
        print("Commit successful.")
        subprocess.run(["git", "push"])
        print("Push successful.")
    else:
        print("Commit aborted.")

if __name__ == "__main__":
    main()

