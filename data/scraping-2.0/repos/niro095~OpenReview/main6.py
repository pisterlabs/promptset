import openai
import os
import git

# Set up OpenAI API key
openai_api_key = input("Enter your OpenAI API key: ")
openai.api_key = openai_api_key

# Set the URL of the GitHub repository to use as input
github_url = input("Enter the URL of the GitHub repository to use as input: ")

# Clone the repository locally
local_repo_path = input("Enter the local directory to clone the repository to: ")
repo = git.Repo.clone_from(github_url, local_repo_path)

# Use OpenAI to optimize and document the code
optimized_code = openai.optimize_code(repo.working_tree_dir)
documented_code = openai.add_documentation(optimized_code)

# Prompt the user for where to push the optimized code
print("Where do you want to push the optimized code?")
print("1. Save to a local folder")
print("2. Push to a new repository")
print("3. Push to the same repository the code came from")
push_destination = input("Enter the number of your selection: ")

if push_destination == "1":
  # Save the optimized and documented code to a new local repository
  output_repo_path = input("Enter the local directory for the output repository: ")
  output_repo = git.Repo.init(output_repo_path)
  output_repo.index.add(documented_code)
  output_repo.index.commit("Optimized and documented code with OpenAI")

elif push_
