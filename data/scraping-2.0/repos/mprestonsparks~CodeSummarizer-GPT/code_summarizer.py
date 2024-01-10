import os
from pathlib import Path
import subprocess
import ast
import json
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CODEBASE_PATH = os.getenv("CODEBASE_PATH")
print(f"Codebase Path: {os.getenv('CODEBASE_PATH')}")

if not os.path.exists('summaries'):
    os.makedirs('summaries')

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).parent / "scripts"
SRC_DIR = Path(CODEBASE_PATH) if CODEBASE_PATH else BASE_DIR
print(f"SRC_DIR: {SRC_DIR}")

ignore_patterns = []
gitignore_path = SRC_DIR / ".gitignore"
if gitignore_path.exists():
    with open(gitignore_path, "r") as f:
        ignore_patterns.extend(line.rstrip('/') if line.endswith('/') else line for line in f.read().splitlines())
ignore_path = BASE_DIR / ".ignore"
if ignore_path.exists():
    with open(ignore_path, "r") as f:
        ignore_patterns.extend(line.rstrip('/') if line.endswith('/') else line for line in f.read().splitlines())

ignore_spec = PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)

import tempfile

def get_components(file_path):
    with open(file_path, "r") as f:
        code = f.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".js") as tmp:
        tmp.write(code.encode())
        tmp_filepath = tmp.name

    try:
        result = subprocess.run(
            ["node", str(SCRIPT_DIR / "parse.js"), str(file_path)],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("An error occurred while running parse.js:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)
        return []

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    os.unlink(tmp_filepath)

    components = json.loads(result.stdout)

    return components

def summarize_code(file_path):
    components = get_components(file_path)

    if components:
        description = f"This file defines the following components: {', '.join(components)}"
    else:
        description = "This file does not define any components."

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"This is a JavaScript file. {description}",
            max_tokens=60,
        )
    except Exception as e:
        print(e)
        return None

    return response['choices'][0]['text'].strip()

def main():
    code_summaries = []
    for root, dirs, files in os.walk(SRC_DIR):
        dirs[:] = [d for d in dirs if not ignore_spec.match_file(os.path.join(root, d))]
        for file in files:
            if file.endswith(".js") or file.endswith(".jsx"):
                full_file_path = os.path.join(root, file)
                if ignore_spec.match_file(full_file_path):
                    continue
                summary = summarize_code(full_file_path)
                relative_file_path = os.path.relpath(full_file_path, str(SRC_DIR))
                print(f"{relative_file_path}:\n\t{summary}")
                code_summaries.append(f"{relative_file_path}:\n\t{summary}")

    with open('summaries/code-summaries.txt', 'w') as f:
        for code_summary in code_summaries:
            f.write(code_summary + '\n\n' + '*'*30 + '\n\n')


if __name__ == "__main__":
    main()
