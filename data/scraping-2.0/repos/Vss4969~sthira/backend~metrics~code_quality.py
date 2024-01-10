"""
Module for code quality metric
"""
import json
from services.openai_service import OpenAIService
from services.github_api import GitHubAPI

openai_service = OpenAIService()
github_api = GitHubAPI()


def parse_file_response(response: dict) -> list[dict]:
    """
    Function to parse the response from the OpenAI API

    Args:
        response (dict): The response from the OpenAI API

    Returns:
        list[dict]: The parsed response
    """
    list_of_dicts = []
    while True:
        try:
            start_index = response.index('{')
            end_index = response.index('}') + 1
            json_substring = response[start_index:end_index]
            response = response[end_index:]
            json_out = json.loads(json_substring)
            if 'file_name' not in json_out or 'sha_hash' not in json_out:
                continue
            list_of_dicts.append(json_out)
        except:  # pylint: disable=bare-except
            break
    return list_of_dicts


def get_files(directory_structure: str) -> list[str]:
    """
    Function to get the files from the directory structure

    Args:
        directory_structure (str): The directory structure

    Returns:
        list[str]: The files in the directory structure
    """
    prompt = """
            You are a Project Manager.
            You will be provided with a directory strucutre tree of a project, where internal nodes 
            are folders and the leaf nodes (without children) are files
            You have to choose ***strictly at max 15 files (you can choose lesser)*** which will be 
            used to analyse the code quality and derive the project description of the given project. 
            Ignore bulk folders like node_modules, .git, etc. Only select sha hases of ***coding***
            files only ignore any file like (node_modules (or similar) folder, excel sheets,
            images, videos, documents, etc.) Only only code files!!
            Format of the output: Make sure you return a a string of filenames and sha hashes. 
            Do not output any string at the beginning or at the end of this as this may be part 
            of an automated evaluation. Keep in mind the syntax of. 
            Do not include any explanations, only provide a  RFC8259 compliant JSON response  
            following this format without deviation.
            The output should be of the form:
            
            [
                {
                    "file_name": <file_name>,
                    "sha_hash": <sha_hash>
                }
            ]
            
            and nothing else.
            
            
            """
    prompt += str(directory_structure)
    response = openai_service.chat(prompt)
    response = response.replace("'", '')
    try:
        response = parse_file_response(response)
    except:  # pylint: disable=bare-except
        print("Error:", response)
    hashes = []
    for data in response:
        try:
            file = data['file_name']
            if file.endswith(".xlsx") or file.endswith(".png") or file.endswith(".jpg"):
                continue
            if file.endswith(".pdf") or file.endswith(".docx") or file.endswith(".txt"):
                continue
            if file.endswith(".mp4") or file.endswith(".mp3") or file.endswith(".avi"):
                continue
            if file.endswith(".pptx") or file.endswith(".ppt") or file.endswith(".doc"):
                continue
            if file.endswith(".csv") or file.endswith(".xls") or file.endswith(".jpeg"):
                continue
            hashes.append(data['sha_hash'])
        except KeyError:
            continue
    return hashes


async def get_code_quality(repo_data: dict, access_token: str, metadata: dict) -> tuple[str, float]:
    """
    Function to get the code quality metric of a repository

    Args:
        repo_data (dict): The repository data
        access_token (str): The GitHub Access Token
        metadata (dict): The metadata of the user

    Returns:
        tuple[str, float]: The code quality description and score
    """
    files = get_files(metadata['directory_structure'])
    metadata['imp_files'] = files
    file_contents = ""
    for file in files:
        try:
            file_content = await github_api.get_file_contents(access_token, repo_data['name'],
                                                              repo_data['owner']['login'], file)
        except Exception as _:  # pylint: disable=broad-except
            continue
        file_contents += file_content + "\n\n"
    prompt = """
            Assume you are a code quality analyzer. 
            You will be provideded with the relevant code files with their names and contents. 
            You are tasked with two things:

            First analyse all the files and come up with a description of the code quality. 
            Be as to the point as possible, be crisp and ***strictly under 4-5 sentences***.
            Enclose the description in a double inverted comma. 

            Second assign a numerical score out of 100 corresponding to the coding evaluation of the 
            project. 
            Ensure the output is ***strictly a single integer***, as it will be stored in an integer 
            format for further evaluation.

            Format of the output: Make sure you provide only a codeblock with json and nothing else. 
            This json will hold a dictionary with two keys, description and score.
            Do not output anything other than the description and the score.
            
            {
                "description": <description>,
                "score": <score>
            }
            
            The concatenated files are given below:

            
            """
    prompt += file_contents
    metadata['file_contents'] = file_contents
    response = openai_service.chat(prompt)
    if response.count('{') == 0 and response.count('}') == 0:
        return "Could not extract description", 0
    try:
        response = json.loads(response)
    except:  # pylint: disable=bare-except
        return "Could not extract description", 0
    return response['description'], response['score']
