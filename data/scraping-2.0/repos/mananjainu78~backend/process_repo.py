import openai
import os
import platform
import subprocess
from clone_repo import extract_user_repo_github
from ast_parser import create_repo_ast
import re
print("ooooo")
def process_repo(url):
    pattern = r'https://github.com/([^/]+)/([^/]+)'
    
    # Search for the pattern in the input URL
    match = re.search(pattern, url)
    print(match)
    if match:
        # Extract the username and repo name from the matched groups
        user_name = match.group(1)
        repo_name = match.group(2)
        if user_name and repo_name:
          print("Username:", user_name)
          print("Repository:", repo_name)
          extract_user_repo_github(url,user_name,repo_name)
          print("nice")
          # create_repo_ast(f"{user_name}_{repo_name}")
          print("ioio")
        else:
          print("Invalid GitHub repository URL")
          return False   
process_repo("https://github.com/soochan-lee/RoT")