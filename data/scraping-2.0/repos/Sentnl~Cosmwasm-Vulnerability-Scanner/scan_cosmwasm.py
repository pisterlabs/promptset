import os
import openai
import fnmatch

def find_rust_files(path):
    rust_files = []

    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.rs'):
            rust_files.append(os.path.join(root, filename))

    return rust_files

def scan_rust_file(file_path, api_key,engine):
    # Connect to the OpenAI API
    openai.api_key = api_key

    # Read the Solidity file content
    with open(file_path) as f:
        rust_code = f.read()

    # Make a request to the OpenAI API to analyze the Solidity file
    result = openai.Completion.create(engine=engine, prompt=f"Analyze vulnerabilities in the following Cosmwasm code:\n{rust_code}\n\nVulnerabilities.When providing your answer provide the line numbers of where the vulnerabilities exist.", max_tokens=1000, n=1, stop=None, temperature=0.5)
    return result.choices[0].text.strip()

def main():
    api_key = os.environ['INPUT_OPENAI_API_KEY']
    engine = os.environ['INPUT_ENGINE']
    repo_path = os.getcwd()
    solidity_files = find_rust_files(repo_path)

    for file in solidity_files:
        print(f"Scanning {file}...")
        vulnerabilities = scan_rust_file(file, api_key, engine)
        with open(f"{file}_vulnerabilities.txt", "w") as f:
            f.write("Service provided to you by www.sentnl.io.\n")
            f.write(f"Vulnerabilities found in {file}:\n{vulnerabilities}\n")

if __name__ == "__main__":
    main()