import json
from collections import deque
import os
from pathlib import Path
import util
import openai
from store.helper import get_or_create_knowledge_from_repo, get_repo_names
from store.store import get_repo_context
from termcolor import colored
from dotenv import load_dotenv, find_dotenv
import subprocess
import gradio as gr
import json

def find_readme(repo_folder):
    # Search for the README file within the found folder
    for filename in os.listdir(repo_folder):
        if filename.lower().startswith('readme'):
            readme_path = os.path.join(repo_folder, filename)
            print("README found in folder:", repo_folder)
            return readme_path

    print("README not found in folder:", repo_folder)
    return None


def bfs_folder_search(text_length_limit=4000, folder_path="./code_repo"):
    if not Path(folder_path).is_dir():
        return "Invalid directory path"

    root = Path(folder_path).resolve()
    file_structure = {str(root): {}}
    queue = deque([(root, file_structure[str(root)])])

    while queue:
        current_dir, parent_node = queue.popleft()
        try:
            for path in current_dir.iterdir():
                if path.is_dir():
                    if str(path.name) == ".git":
                        continue
                    parent_node[str(path.name)] = {"files": []}
                    queue.append((path, parent_node[str(path.name)]))
                else:
                    if "files" not in parent_node:
                        parent_node["files"] = []
                    parent_node["files"].append(str(path.name))

                # Check if we've exceeded the text length limit
                file_structure_text = json.dumps(file_structure)
                if len(file_structure_text) >= text_length_limit:
                    return file_structure

        except PermissionError:
            # This can happen in directories the user doesn't have permission to read.
            continue

    return file_structure


def get_repo_structure(code_repo_path="./code_repo"):
    return bfs_folder_search(4000, code_repo_path)


def find_repo_folder(directory):
    # Find the name of the first folder in the specified directory
    folder_name = None
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_name = item
            break
    return os.path.join(directory, folder_name)


# Find the Readme.md file from the code repo in the code_repo folder
# summarize the README file
def summarize_readme(readme_path):
    if readme_path:
        print(colored("Summarizing README...", "green"))

        system_prompt = """You are an expert developer and programmer. 
            Please infer the programming languages from the README.
            You are asked to summarize the README file of the code repository in detail. 
            Provide enough information about the code repository.
            Please also mention the framework used in the code repository.
            """
        readme_content = open(readme_path, "r").read()
        user_prompt = f'Here is the README content: {readme_content}'
        return util.get_chat_response(system_prompt, user_prompt)


def get_readme_summary(code_repo_path="./code_repo"):
    repo_folder = find_repo_folder(code_repo_path)
    print(colored("Repo folder: " + repo_folder, "green"))
    readme_path = find_readme(repo_folder)
    if readme_path is None:
        return "README not found"
    else:
        summary = summarize_readme(readme_path)
        print(colored("README Summary: ", "green"), colored(summary, "green"))
        return summary


def clone_repo(git_url, progress=gr.Progress(), code_repo_path="./code_repo"):
    print(progress(0.1, desc="Cloning the repo..."))
    print("Cloning the repo: ", git_url)
    # Check if directory exists
    if not os.path.exists(code_repo_path):
        os.makedirs(code_repo_path)
    try:
        subprocess.check_call(['git', 'clone', git_url], cwd=code_repo_path)
        print(f"Successfully cloned {git_url} into {code_repo_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")

    print(progress(0.3, desc="Summarizing the repo..."))
    readme_info = get_readme_summary(code_repo_path)
    if readme_info is not None:
        readme_info = """The README.md file is as follows: """ + readme_info + "\n\n"

    print(progress(0.4, desc="Parsing repo structure..."))
    repo_structure = get_repo_structure(code_repo_path)
    if repo_structure is not None:
        repo_structure = """The repo structure is as follows: """ + get_repo_structure(code_repo_path) + "\n\n"

    return readme_info + repo_structure


if __name__ == '__main__':
    code_repo_path = "./code_repo"
    load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY", "null")

    print(get_repo_names(code_repo_path))

    # Basic repo information
    get_readme_summary(code_repo_path)
    print(colored(json.dumps(bfs_folder_search(4000, code_repo_path)), "yellow"))

    # Generate knowledge base
    vdb = get_or_create_knowledge_from_repo("./code_repo")

    # Search the knowledge base
    query = "How to use the knowledge base?"
    context = get_repo_context(query, vdb)
    print(context)
