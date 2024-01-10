"""
Module to get the commits metric.
"""
from services.github_api import GitHubAPI
from services.openai_service import OpenAIService

openai_service = OpenAIService()
github_api = GitHubAPI()


async def get_commits_metric(repo_data: dict, access_token: str, metadata: dict) -> str:
    """
    Function to get the commits descriptions of a repository

    Args:
        repo_data (dict): The repository data
        access_token (str): The GitHub Access Token
        metadata (dict): The metadata of the user

    Returns:
        str: The commits description of the repository
    """
    commits = await github_api.get_commits(access_token, repo_data['name'],
                                           repo_data['owner']['login'])
    if commits is None:
        return "No commits found!"

    prompt = """
            You are a Project Manager.
            You will be provided with a project description of a project and the list of the commits 
            in order from the start of the project until now sorted in time.
            
            Analyse the whole commits list against the project descirption and come up with a 
            description for the quality of the commits.

            Make sure you provide you do not output anything other than the description.
            Keep the description as to the point as possible, 
            be crisp and ***strictly under 2-3 sentences***.
            Enclose the description in a double inverted comma.

            Project Description:
            """

    prompt += metadata['project_descriptions']
    prompt += """
                List of commits:
                """
    prompt += commits
    response = openai_service.chat(prompt)
    return response
