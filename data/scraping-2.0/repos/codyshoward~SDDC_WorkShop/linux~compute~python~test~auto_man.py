import os
import hashlib
import logging
import openai

# Configure logging
logging.basicConfig(filename='script.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# OpenAI GPT-3 API credentials
api_key = "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"
client = openai.Client(api_key=api_key)

# GitHub authentication
github_token = "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"
github_username = "codyshoward"
github_pat = "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"  # Replace with your GitHub PAT
github_email = "codyshoward@gmail.com"
# GitHub Repo
repo_name = "SDDC_WorkShop"

# Function to generate a man file for a script file
def generate_man_file(script_file):
    with open(script_file, "r") as f:
        script_content = f.read()

    try:
        response = client.Completion.create(
            engine="text-davinci-002",
            prompt=f"Generate a man page for the script or module in {script_file}:\n\n{script_content}\n\n",
            max_tokens=200
        )
        man_content = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in generating man page: {e}")
        return

    man_file = script_file + ".man"
    with open(man_file, "w") as f:
        f.write(man_content)

def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64 KB chunks
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()

# Initialize a list to store script files that need man pages
script_files_to_generate = []

def check_and_update_man_pages(repo_name):
    print(f"Cloning repository {repo_name}...")
    repo_dir = repo_name
    clone_result = os.system(f'git clone https://github.com/{github_username}/{repo_name}.git {repo_dir}')
    if clone_result != 0:
        print("Failed to clone the repository.")
        return

    os.chdir(repo_dir)
    logging.info(f"Changed directory to {os.getcwd()}")
    print(f"Changed directory to {os.getcwd()}")

    current_hashes = {}
    script_files_to_generate.clear()

    if os.path.exists("../hashfile.txt"):
        with open("../hashfile.txt", "r") as f:
            for line in f:
                file_hash, file_path = line.strip().split(":")
                current_hashes[file_path] = file_hash

    for root, dirs, files in os.walk("."):
        print(f"Searching in directory: {root}")
        for file in files:
            if file.endswith((".sh", ".py", ".yaml", ".yml", ".ps")):
                script_file = os.path.join(root, file)
                print(f"Processing file: {script_file}")
                file_hash = calculate_file_hash(script_file)
                if current_hashes.get(script_file) != file_hash:
                    script_files_to_generate.append(script_file)
                    current_hashes[script_file] = file_hash

    with open("../hashfile.txt", "w") as f:
        for file_path, file_hash in current_hashes.items():
            f.write(f"{file_hash}:{file_path}\n")

    print("File analysis completed.")
    for script_file in script_files_to_generate:
        generate_man_file(script_file)
        print(f"Man page generated for {script_file}")

    print("Manual files have been created.")

if __name__ == "__main__":
    check_and_update_man_pages(repo_name)
