import os
import git
from openai import OpenAI

# Create an OpenAI instance
openai = OpenAI()

# Get the GitHub repository URL
repo_url = input("Please enter the GitHub repository URL: ")

# Clone the repository
repo = git.Repo.clone_from(repo_url, "local_repo")

# Optimize the code
optimized_code = openai.optimize_code(repo.working_dir)

# Save the optimized code to a file
with open("optimized_code.py", "w") as f:
    f.write(optimized_code)

# Commit the optimized code to the repository
repo.index.add("optimized_code.py")
repo.index.commit("Optimized code")

# Push the changes to the remote repository
repo.remote().push()

# Clean up
os.remove("optimized_code.py")
