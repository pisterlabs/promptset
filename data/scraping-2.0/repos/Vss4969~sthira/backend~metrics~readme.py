"""
Module for Reame file analysis
"""

from services.github_api import GitHubAPI
from services.openai_service import OpenAIService

openai_service = OpenAIService()
github_api = GitHubAPI()


async def get_readme_metric(repo_data: dict, access_token: str, metadata: dict) -> str:
    """
    Function to get the readme file analysis

    Args:
            repo_data (dict): The repository data
            access_token (str): The GitHub Access Token
            metadata (dict): The metadata of the user

     Returns:
            str: The description of the readme file
    """
    readme = await github_api.get_readme(access_token, repo_data['name'],
                                         repo_data['owner']['login'])
    if readme is None:
        return "No Readme content found!"
    prompt = """
            You are a Project Manager. You will be provided with a project description of a project 
            and the content of readme file of the project.
            
            Analyse the readme file against the project descirption and come up with a description 
            for the quality of the readme file.

            Make sure you provide you don not output anything other than the description. Keep the 
            description as to the point as possible, be crisp and ***strictly under 1-2 sentences***
            Enclose the description in a double inverted comma.

            Project Description:
            """

    # MAKE SURE TO KEEP THE PROJECT DESCRIPTION IN "metadata['description']"
    prompt += metadata['project_descriptions']
    prompt += """
            
            Readme Content:
            """
    prompt += readme
    response = openai_service.chat(prompt)
    return response
