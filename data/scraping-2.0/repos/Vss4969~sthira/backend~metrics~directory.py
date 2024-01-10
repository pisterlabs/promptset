"""
Module to analyse complete directory structure of a repository
"""

import copy
from services.github_api import GitHubAPI
from services.openai_service import OpenAIService

github_api = GitHubAPI()
openai_service = OpenAIService()
# pylint: disable=global-variable-not-assigned,global-at-module-level
global full_tree


def is_folder(node: dict) -> bool:
    """
    Returns True if the node is a folder

    Args:
        node (dict): The node to check

    Returns:
        bool: True if the node is a folder
    """
    return node['type'] == 'dir' or node['type'] == 'tree'


async def construct_directory_structure(root_structure: dict, access_token: str,
                                        repo_name: str, owner: str) -> dict:
    """
    Function to construct the directory structure

    Args:
        root_structure (dict): The root structure of the directory
        access_token (str): The GitHub Access Token
        repo_name (str): The repository name
        owner (str): The owner of the repository

    Returns:
        dict: The directory structure
    """
    directory_structure = {}
    for node in root_structure:
        if not is_folder(node):
            directory_structure[(node['path'], node['sha'])] = {}
        else:
            folder_data = await github_api.get_folder_contents(access_token, repo_name, owner,
                                                               node['sha'])
            folder_data = folder_data['tree']
            directory_structure[(node['path'], node['sha'])] = await construct_directory_structure(
                folder_data, access_token, repo_name, owner)
    return directory_structure


def stringed_directory(directory_structure: dict, depth: int = 0) -> None:
    """
    Function to convert the directory structure to a string and store in global variables
    full_tree

    Args:
        directory_structure (dict): The directory structure
        depth (int, optional): The depth of the current node. Defaults to 0.
    """
    # pylint: disable=global-variable-undefined
    global full_tree
    for key, value in directory_structure.items():
        full_tree += '\t' * depth + f'-- {key[0]} {key[1]}\n'
        if len(value):
            stringed_directory(value, depth + 1)


async def get_directory_metric(repo_data: dict, access_token: str, metadata: dict) -> str:
    """
    Function to get the directory metric from the directory structure

    Args:
        repo_data (dict): The repository data
        access_token (str): The GitHub Access Token
        metadata (dict): The metadata of the user

    Returns:
        str: The description of the directory structure
    """
    # pylint: disable=global-variable-undefined
    global full_tree
    full_tree = ""
    root_struct = await github_api.get_repository_contents(access_token, repo_data['name'],
                                                           repo_data['owner']['login'])
    full_structure = await construct_directory_structure(root_struct, access_token,
                                                         repo_data['name'],
                                                         repo_data['owner']['login'])
    stringed_directory(full_structure)
    metadata['directory_structure'] = copy.deepcopy(full_tree)
    prompt = """
            Assume that you are a GitHub Repo analyzer for GitHub. You have to take in a Directory
            structure as a dictionary (name, sha) and output 2-3 sentnces . Be as to the point as
            possible, be crisp and only 2-3 sentences. The directory Structure is shown below. 
            Enclose the comments in a double inverted comma. Do not output anything other than the 
            comments. Comment on File and Folder naming convention, code modularization, etc.
            Detect if the structure are too simple or not ordered. The directory structure is:
            
            
            """
    prompt += full_tree
    response = openai_service.chat(prompt)
    return response
