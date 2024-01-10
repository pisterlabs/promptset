#!/usr/bin/python3
import os
import openai
import sys
import subprocess
import json

gpt_key = os.environ.get("OPEN_AI_KEY")
openai.api_key = gpt_key

args = sys.argv[1:]


IGNORED_FILES = [
    "pnpm.lock",
    "yarn.lock",
    "package-lock.json",
]


def get_commit_message_from_gpt(file_content):
    """
    Generates a semantic commit message using OpenAI's GPT-3.5-turbo model.

    Args:
        file_content (str): The content of the file.

    Returns:
        str: The generated commit message, including the type of the commit.

    Example Usage:
        >>> file_content = "This is the content of the file."
        >>> get_commit_message_from_gpt(file_content)
        "feat: add a new file"

    Note:
        - The file_content should be less than 50 words.
        - The commit message will only contain the commit message and the type of the commit, e.g., "feat: add a new file".

    Raises:
        ValueError: If the file_content is empty or exceeds the word limit.

    """
    prompt = """Generate a semantic commit message for next files, commit message should be less than 50 words,  ex: feat: add a new file 
     
    based on this information, create a commit message, a commit description for this diffs of this files
    
    the return should be exactly like this example:


    {
        "message": "feat: add a new file",
        "description": add a new file to the project
    }
    
    """

    if not file_content:
        raise ValueError("The file_content is empty")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": file_content.replace("\n", " ").strip()},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    return response.choices[0].message.content


def get_git_diff_by_files(folder_path):
    """
    Given a folder path, this function retrieves the git diff for each file in the folder.

    Parameters:
        folder_path (str): The path of the folder containing the git repository.

    Returns:
        dict: A dictionary where the keys are the file names and the values are the git diffs.

    Raises:
        ValueError: If the given directory is not a git repository.
    """
    os.chdir(folder_path)

    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"])
    except subprocess.CalledProcessError:
        raise ValueError("The given directory is not a git repository")

    diff_output = subprocess.check_output(["git", "diff"], text=True)

    file_diffs = {}
    current_file = None
    IGNORED_FILES = set()

    for line in diff_output.split("\n"):
        if line.startswith("diff"):
            if current_file and (current_file not in IGNORED_FILES):
                file_diffs[current_file] = "\n".join(file_content)
            current_file = line.split()[-1]
            file_content = [line]
        elif current_file:
            file_content.append(line)

    if current_file:
        file_diffs[current_file] = "\n".join(file_content)

    return file_diffs


folder_path = args[0] if len(args) > 0 else os.getcwd()
diff_content = get_git_diff_by_files(folder_path)

full_content = ""
for file_name, file_content in diff_content.items():
    full_content += file_content


commit_message = get_commit_message_from_gpt(full_content)
print(commit_message)

json_content = json.loads(
    commit_message
)

os.system("git add .")
os.system("git commit -m \"{}\" -m \"{}\"".format(json_content['message'], json_content['description']))