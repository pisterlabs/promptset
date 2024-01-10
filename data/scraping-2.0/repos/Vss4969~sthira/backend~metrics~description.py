"""
Module to analyse the description of a repository
"""

from services.openai_service import OpenAIService


openai_service = OpenAIService()


def get_repo_description(metadata: dict) -> str:
    """
    Function to get the description of a repository

        Args:
                metadata (dict): The metadata of the user

        Returns:
                str: The description of the repository
    """
    prompt = """
            Assume that you are a repo analyzer for GitHub. You have to take in some code files
            from the reposiotry concatenated together and output 2-3 sentnces ONLY about the repo
            and a description of what is the project about. Be as to the point as possible, be crisp
            and only 2-3 sentences. the code is shown below. Enclose the comments in a double 
            inverted comma. Do not output anything other than the comments. The code is given below:
            
            """
    prompt += metadata['file_contents']
    response = openai_service.chat(prompt)
    metadata['project_descriptions'] = response
    return response
