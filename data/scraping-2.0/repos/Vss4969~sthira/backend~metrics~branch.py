"""
Module to get branch metric
"""

from services.github_api import GitHubAPI
from services.openai_service import OpenAIService


github_api = GitHubAPI()
openai_service = OpenAIService()


async def get_branch_metric(repo_data: dict, access_token: str, metadata: dict) -> str:
    """
    Function to get the branch metric of a repository

    Args:
        repo_data (dict): The repository data
        access_token (str): The GitHub Access Token
        metadata (dict): The metadata of the user

    Returns:
        str: The branch metric of the repository
    """
    branches = await github_api.get_branches(access_token, repo_data['name'],
                                             repo_data['owner']['login'])
    metadata['branches'] = branches
    if len(branches) <= 1:
        return "Only Master/Main branch Exists"
    prompt = """
            Assume that you are a branch analyzer for GitHub. You have to take in a List of branhces
            and output 2-3 sentnces ONLY about the branches and branch names. Be as to the point as
            possible, be crisp and only 2-3 sentences. the branch data is shown below. Enclose the
            comments in a double inverted comma. Do not output anything other than the comments.
            Also give a comment on the naming convention (including other features of the branch 
            not only name) of the branches. Detect if the names are too simple or not enough 
            branches. The branches are:
            
            
            """
    prompt += str(branches)
    response = openai_service.chat(prompt)
    return response, branches
