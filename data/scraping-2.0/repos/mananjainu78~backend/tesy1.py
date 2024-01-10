import openai
import os
import platform
import subprocess
from tree_sitter import Language, Parser
import re
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
def extract_user_repo_github(url):
    # Define a regular expression pattern to match GitHub repository URLs
    pattern = r'https://github.com/([^/]+)/([^/]+)'
    
    # Search for the pattern in the input URL
    match = re.search(pattern, url)
    
    if match:
        # Extract the username and repo name from the matched groups
        user_name = match.group(1)
        repo_name = match.group(2)
        if user_name and repo_name:
          print("Username:", user_name)
          print("Repository:", repo_name)
        else:
          print("Invalid GitHub repository URL")
          return False
        destination = f"backend/repositories/{user_name}_{repo_name}"
        create_folder(destination)  
        try:
        # Run the 'git clone' command
          result = subprocess.run(["git", "clone", url, destination], capture_output=True, text=True, check=True)
        # 'check=True' raises a CalledProcessError if the command exits with a non-zero status
        # 'capture_output=True' captures the standard output and standard error

        # Print the output
          print("Clone Output:", result.stdout)
        
        except subprocess.CalledProcessError as e:
        # An error occurred, print the error message
          print("Error cloning repository nopes:", e.stderr)
          return False
    
        return True

# # Example usage
# url = "https://github.com/soochan-lee/RoT"
# clone_return_status = extract_user_repo_github(url)
# print(clone_return_status)


current_dir = os.getcwd()
print("Current working directory:", current_dir)
from pathlib import Path

current_dir = Path.cwd()
print("Current working directory:", current_dir)
plat = platform.system()     
print(plat)

root = Path(__file__).parent

from tree_sitter import Language, Parser
language=['python','java','javascript','cpp','rust']
for lang in language:
  grammar_repo_path = os.path.join(root, "tree_sitter_grammar", f"tree-sitter-{lang}")
  grammar_output_path = os.path.join(root, "tree_sitter_grammar",f"{lang}.so")
  print(grammar_repo_path)
  Language.build_library(
  # Store the library in the `build` directory
  grammar_output_path,

  # Include one or more languages
  [
    grammar_repo_path,

  ]
)



# current_directory = r'C:\Users\Tanmay Saini\Desktop\codeconverse\backend\tree_sitter_grammar'
# print("dfgdfg "+current_directory)

# def clone_repository(local_path):
#     """Clone the specified git repository to the given local path."""
#     subprocess.run(["git", "clone", "https://github.com/tree-sitter/tree-sitter-python", local_path])

# clone_repository(current_directory)



x={}
def ac():
   if "qwer" in x:
      print("oho")
   x["qwer"]=1
   
   if "qwer" in x:
      print("noikl")
