# # Introduction
# Introduce the topic of retrieving file contents from a GitHub repository using the GitHub API and provide an overview of the notebook.

# Introduction
#
# This Jupyter notebook demonstrates how to retrieve file contents from a GitHub repository using the GitHub API.
# We will be using the requests library to make HTTP requests to the GitHub API and the json library to parse the responses.
# The notebook is divided into the following sections:
# 1. Introduction (this section)
# 2. Authenticating with the GitHub API
# 3. Retrieving file contents from a GitHub repository
# 4. Conclusion and next steps

import openai
import requests # for making HTTP requests
import json # for parsing responses
import os
import base64
import json
import subprocess

api_endpoint = "https://api.github.com"
api_key = os.getenv("GITHUB_API_TOKEN")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/vnd.github.v3+json"
}
def get_remote_files(path):
    print(path)
    if not path.startswith('http'):
        file_content = open(path, encoding='UTF-8').read()
    # url = https://github.com/gitkraken/vscode-gitlens/blob/0c3c28df26813733cc74cced0aec1ffc89151518/src/ai/openaiProvider.ts#L71
    repo = "gitkraken"
    owner = "vscode-gitlens"
    # path = "src/ai/openaiProvider.ts"
    # path = '/'.join(url.split('#')[0].split('/')[-3:])
    print('repo:',repo,' owner:',owner,' path:',path)
    url = f"{api_endpoint}/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, headers=headers)
    # decode the response
    content = json.loads(response.text)
    # decode the content
    file_content = base64.b64decode(content["content"]).decode("utf-8")
    return file_content

import os
import subprocess

def is_git_controlled(item_path):
    """
    Check if a directory or file is under Git control.
    """
    try:
        if os.path.isfile(item_path):
            cwd = os.path.dirname(item_path)
            filename = os.path.basename(item_path)
            output = subprocess.check_output(['git', 'ls-files', filename], cwd=cwd).decode().strip()
            return output == filename
        else:
            subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'], cwd=item_path)
            return True
    except subprocess.CalledProcessError:
        return False
def is_folder_excluded(item_path):
    excluded_folder = ['image', 'images', 'docs', 'doc', 'static', 'assets', 'dist', 'build', 'node_modules', 'vendor','__pycache__']
    if item_path.split('/')[-1] in excluded_folder:
        print("excluded", item_path)
        return True

def is_file_excluded(item_path):
    excluded_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.fig', '.ai', '.psd', '.pdf', '.zip', '.rar', '.gz', '.tar', '.7z', '.mp4', '.mp3', '.wav']
    excluded_folder = ['image', 'images', 'docs', 'doc', 'static', 'assets', 'dist', 'build', 'node_modules', 'vendor','__pycache__']

    excluded_file = ['.gitignore', '.gitattributes', '.DS_Store', '.gitkeep', '.gitmodules', '.git', '.github', '.vscode', '.idea', '.gitpod', '.gitpod.yml', '.gitpod.Dockerfile', 'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'LICENSE.rst', 'LICENSE.json', 'LICENSE.xml', 'LICENSE.html', 'LICENSE.yml', 'LICENSE.yaml', 'LICENSE.ini', 'LICENSE.toml', 'LICENSE.php', 'LICENSE.tpl', 'LICENSE.txt.sample', 'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml', 'docker-compose.dev.yml', 'docker-compose.dev.yaml', 'docker-compose.prod.yml', 'docker-compose.prod.yaml', 'docker-compose.test.yml', 'docker-compose.test.yaml', 'docker-compose.override.yml', 'docker-compose.override.yaml', 'docker-compose.ci.yml', 'docker-compose.ci.yaml', 'docker-compose.override.ci.yml', 'docker-compose.override.ci.yaml', 'docker-compose.override.prod.yml', 'docker-compose.override.prod.yaml', 'docker-compose.override.dev.yml', 'docker-compose.override.dev.yaml', 'docker-compose.override.test.yml', 'docker-compose.override.test.yaml']
    only_one_ReadMe = False

    # if not is_git_controlled(item_path):
    #     return True

    for extension in excluded_extensions:
        if item_path.endswith(extension):
            print("excluded", item_path)
            return True
    # for keyword in excluded_keywords:
    #     if keyword in item_path:
    #         print("excluded", item_path)
    #         return True
    for file in excluded_file:
        if item_path.endswith(file):
            print("excluded", item_path)
            return True
    if item_path.endswith('README.md'):
        if only_one_ReadMe:
            print("excluded", item_path)
            return True
        else:
            only_one_ReadMe = True
    return False
def save_schema(file_contents,path):
    schema = {
        "System": {
            "description": "type of string",
            "rules": [
                "string 0",
                "string 1",
                "string 2",
                "string 3",
                "string 4",
            ],
            "outputFormat": {
                "projectOutline": {
                    "repositoryUrl": path,
                    "filePaths": [],
                },
                "projectFiles": {
                    "files": []
                }
            }
        }
    }
    schema["System"]["outputFormat"]["projectFiles"]["files"] = file_contents
    with open('schema.json', 'w') as f:
        json.dump(schema, f, indent=4)
    return schema

# def get_directory_tree(local_path, only_path=False,need_to_check_if_git_controled=True):
#     """
#     Recursively retrieve the contents of a directory and its subdirectories that are under Git control.
#     """
#     import questionary
#     print(f"Entering get_directory_tree with local_path = {local_path} and only_path = {only_path}")
#     import chardet

#     def detect_encoding(file_path):
#         with open(file_path, 'rb') as f:
#             result = chardet.detect(f.read())
#         return result['encoding']
#     # Send GET request to retrieve directory contents
#     contents = os.listdir(local_path)

#     # Initialize list to hold file contents
#     file_contents = []

#     # Iterate through directory contents
#     for item in contents:
#         item_path = os.path.join(local_path, item)
#         print(f"Processing item: {item_path}, is_git_controlled: {is_git_controlled(item_path)}, is_file_excluded: {is_file_excluded(item_path)}")
#         if(need_to_check_if_git_controled):
#             if not is_git_controlled(item_path):
#                 print(f"Skipping item {item_path} because it is not under Git control")
#                 continue
#         if is_file_excluded(item_path):
#             print(f"Skipping item {item_path} because it is excluded")
#         if os.path.isdir(item_path):
#                 if '.git' in item_path:
#                     continue
#                 if is_folder_excluded(item_path):
#                     continue
#                 if not is_git_controlled(item_path):
#                     continue
#                 # Recursively retrieve contents of subdirectory
#                 print(f"Processing subdirectory: {item_path}")
#                 subdirectory_contents = get_directory_tree(item_path,only_path=only_path)
#                 file_contents.extend(subdirectory_contents)
#         else:
#             # Retrieve contents of file
#             print(f"Processing file: {file_path} with encoding: {encoding}")
#             file_path = item_path
#             encoding = detect_encoding(file_path)
#             if only_path:
#                 file_contents.append(file_path)
#             else:
#                 file_content = open(file_path, encoding=encoding, errors='ignore').read()
#                 print(file_path)
#                 file_contents.append({
#                     "path": file_path,
#                     "content": file_content,
#                     "isValid": True,  # You can set this based on your validation logic
#                 })
#     return save_schema(file_contents,local_path)
import os
import chardet

class DirectoryTree:
    def __init__(self, local_path, only_path=False):
        self.local_path = local_path
        self.only_path = only_path
        self.file_contents = []
        self.need_to_check_if_git_controled = False
        if os.path.isdir(local_path):
            if '.git' in local_path:
                self.need_to_check_if_git_controled = True
                # self.need_to_check_if_git_controled = True

    def _is_git_controlled(self, path):
        if not self.need_to_check_if_git_controled:
            return True
        return is_git_controlled(path)
    def _is_file_excluded(self, path):
        # Implement your logic to check if a file should be excluded
        return is_file_excluded(path)

    def get_file_content(self, path):
        with open(path, 'rb') as f:
            rawdata = f.read()

        encoding = chardet.detect(rawdata)['encoding']
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        return content, encoding

    def process_file(self, path):
        print(f"Processing file: {path}")
        content, encoding = self.get_file_content(path)
        self.file_contents.append({
            'path': path,
            'content': content if not self.only_path else '',
            'encoding': encoding,
            'isValid': True  # Add your validation logic here
        })

    def get_directory_tree(self, local_path=None):
        if local_path is None:
            local_path = self.local_path

        for item in os.listdir(local_path):
            item_path = os.path.join(local_path, item)

            if os.path.isfile(item_path):
                if not self._is_git_controlled(item_path) or self._is_file_excluded(item_path):
                    print(f"Skipping item {item_path} because it is not under Git control or is excluded")
                    continue
                self.process_file(item_path)

            elif os.path.isdir(item_path):
                if is_folder_excluded(item_path):
                    continue
                self.get_directory_tree(item_path)
        return save_schema(self.file_contents,local_path)
def get_local_json(path,only_path=False, rules=None):
    # using appid to query sql files to get a directory tree
    dir_tree = DirectoryTree(path,only_path)
    return dir_tree.get_directory_tree()
    # return get_directory_tree(path,only_path,need_to_check_if_git_controled)
def get_repo_json(url,verbose=False):
    schema_contents = []
    owner = url.split('/')[-2]
    repo = url.split('/')[-1]

    # Construct the API endpoint URL
    url = f"{api_endpoint}/repos/{owner}/{repo}/contents"

    # Make the API request
    response = requests.get(url, headers=headers)

    # Parse the response JSON
    response_json = json.loads(response.text)

    # Initialize the schema structure
    for i in response_json:
        print(i)
    response_json[0]['name']
    # Define the schema for projectFiles
    project_files_schema = lambda file_path, file_content: {
        "name": file_path.split('/')[-1],
        "path": file_path,
        "content": file_content,
        "isValid": True,  # You can set this based on your validation logic
    }
    # Extract the file path and content from the JSON data
    file_path = response_json[0]['path']

    def get_directory_contents(path):
        """
        Recursively retrieve the contents of a directory and its subdirectories.
        """
        # Send GET request to retrieve directory contents
        url = f"{api_endpoint}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        contents = json.loads(response.text)

        # Initialize list to hold file contents
        file_contents = []

        # Iterate through directory contents
        for item in contents:
            if is_file_excluded(item['path']):
                continue
            if item['type'] == 'dir':
                # Recursively retrieve contents of subdirectory
                subdirectory_contents = get_directory_contents(item['path'])
                file_contents.extend(subdirectory_contents)
            else:
                # Retrieve contents of file
                file_path = item['path']
                print(file_path)
                url = f"{api_endpoint}/repos/{owner}/{repo}/contents/{file_path}"
                response = requests.get(url, headers=headers)
                content = json.loads(response.text)
                try:
                    file_content = base64.b64decode(content["content"]).decode("utf-8")
                except Exception as e:
                    # raise Exception(f"Error decoding file content for {file_path}: {e}")
                    file_content = ""

                # Add file content to list
                file_contents.append(project_files_schema(file_path, file_content))

        return file_contents

    new_json = []
    for file in response_json:
        if is_file_excluded(file['path']):
            continue
        if '.github' in file['path']:
            continue
        if file['type'] == 'dir':
            new_json.extend(get_directory_contents(file['path']))
            continue
        file_path = file['path']
        print(file_path)
        url = f"{api_endpoint}/repos/{owner}/{repo}/contents/{file_path}"
        response = requests.get(url, headers=headers)
        content = json.loads(response.text)
        file_content = base64.b64decode(content["content"]).decode("utf-8")

       

        # Add the project file schema to the schema structure
        schema_contents.append(project_files_schema(file_path, file_content))
    schema_contents.extend(new_json)
    return save_schema(schema_contents,url)

def schema_json_extract_only_path(schema):
    with open(schema) as f:
        data = json.load(f)
    paths = []
    for idx, _ in enumerate(data['System']['outputFormat']['projectFiles']['files']):
        data['System']['outputFormat']['projectFiles']['files'][idx].pop('content', None)
    # copy to the clipboard
    pyperclip.copy(json.dumps(data))


GET_REPO_FILES = {
    'definitions': [
        {'name': 'get_file_content',
            'description': 'Get the file content from project.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': 'path of file in the project',
                    },
                }
            },
        },
        {'name': 'questions_clarification',
            'description': 'ask user to clarify the question',
            'parameters': {
                'type': 'object',
                'properties': {
                    'clarifications': {
                        'type': 'array',
                        'description': 'clarification questions',
                        'items': {
                            'type': 'string',
                            'description': 'clarification question',
                        }
                    }
                }
            },
        },
    ],
    'functions': {
        'get_file_content': lambda path: open(path, encoding='UTF-8').read(),
        'questions_clarification': lambda clarifications: clarifications,
    }
}