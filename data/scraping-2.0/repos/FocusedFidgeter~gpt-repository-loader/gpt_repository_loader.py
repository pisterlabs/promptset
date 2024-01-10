#!/usr/bin/env python3

"""
This module provides functionality for analyzing and loading code
    from a Git repository or directory.
The purpose of this module is to provide a tool for generating
    a markdown document outlining the file structure and
    providing explanations for the code in a given Git repository
    or directory.
"""

import os
import sys
import fnmatch
from datetime import datetime
from pathlib import PurePath
from openai import OpenAI

# ----------------------------------- Helper Functions ----------------------------------- #

def get_ignore_list(ignore_file_path):
    """
    Generates a list of patterns to ignore based on
        the contents of the specified ignore file.

    Args:
        ignore_file_path (str): The path to the ignore file.

    Returns:
        list: A list of patterns to ignore.
    """
    ignore_list = []
    with open(ignore_file_path, "r", encoding="utf-8") as ignore_file:
        for line in ignore_file:
            if sys.platform == "win32":
                line = line.replace("/", "\\")
            ignore_list.append(line.strip())
    return ignore_list


def should_ignore(file_path, ignore_list):
    """
    Check if a file should be ignored based on a list of patterns.

    Args:
        file_path (str): The path of the file to check.
        ignore_list (list): A list of patterns to match against the file path.

    Returns:
        bool: True if the file should be ignored, False otherwise.
    """
    for pattern in ignore_list:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False

def chunk_code(code, chunk_size):
    """
    Generates chunks of code from a given code string.

    Args:
        code (str): The code string to be chunked.
        chunk_size (int): The number of lines in each chunk.

    Yields:
        str: A chunk of code from the given code string.
    """
    code_lines = code.splitlines()
    for i in range(0, len(code_lines), chunk_size):
        yield "\n".join(code_lines[i: i + chunk_size])

def write_to_file(output_file, relative_file_path, chunk, explanation):
    """
    Writes the given chunk of code and its explanation to the output file.

    Parameters:
        output_file (file): The file object to write the chunk of code and explanation to.
        relative_file_path (str): The relative path of the file.
        chunk (str): The chunk of code to be written.
        explanation (str): The explanation of the code.

    Returns:
        None
    """
    output_file.write("-" * 3 + "\n")
    output_file.write(f"\n# {relative_file_path}\n")
    output_file.write(f"\n## Code:\n```\n{chunk}\n```\n")
    output_file.write(f"\n## Explanation:\n{explanation}\n")


def get_code_explanation(chunk, client, model):
    """
    Retrieves a code explanation for the given code chunk.

    Args:
        chunk (str): The code chunk for which an explanation is needed.
        client (str): The client used for making the API call.
        model (str): The model used for generating the code explanation.

    Returns:
        str: The generated code explanation.
    """
    prompt = f"""
        Write a brief explanation of the following code.
        A quick 1-2 sentence summary of the main functions or blocks of code.
        This explanation should be written mainly for new developers for
            understanding purposes: \n\n{chunk}
        """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant providing code explanations.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
        max_tokens=3000,
    )
    explanation = response.choices[0].message.content.strip()
    return explanation


def get_file_size(path):
    """
    Fetches the file size in bytes.

    Args:
        path (str): The path to the file.

    Returns:
        int: The size of the file in bytes.
    """
    return os.path.getsize(path)


def get_last_modified(path):
    """
    Fetches the file's last modified time.

    Args:
        path (str): The path of the file.

    Returns:
        str: The last modified time in ISO format.
    """
    return datetime.fromtimestamp(os.path.getmtime(path)).isoformat()

# ----------------------------------- Logic Functions ----------------------------------- #

def create_file_structure(path, md_file_path, indent="    "):
    """
    Creates markdown documentation for the structure of a given directory path.

    Args:
        path (str): The path to the directory.
        md_file_path (str): The output markdown file path.
        indent (str, optional): The indentation for the directory structure. Default is '    '.

    Returns:
        None
    """
    with open(md_file_path, "a", encoding="utf-8") as file:
        file.write(indent + "- Directory: " + PurePath(path).name + "\n")

    if os.path.isdir(path):
        for inner_item in os.listdir(path):
            inner_path = os.path.join(path, inner_item)

            if os.path.isdir(inner_path):
                create_file_structure(
                    inner_path, md_file_path, indent=indent + "    ")
            else:
                with open(md_file_path, "a", encoding="utf-8") as file:
                    file_size = get_file_size(inner_path)
                    last_modified = get_last_modified(inner_path)
                    file.write(indent + "    " + f"- File: {PurePath(inner_path).name}, Size: {file_size} bytes, Last Modified: {last_modified}\n")

def process_repository(repo_path, ignore_list, output_file):
    """
    Process a repository by reading the files in a given directory and
        generating explanations for the code chunks.

    Args:
        repo_path (str): The path to the repository directory.
        ignore_list (list): A list of file patterns to ignore.
        output_file (_io.TextIOWrapper): The output file to write the explanations to.

    Returns:
        None
    """
    model = "gpt-3.5-turbo-1106"  # specify the model
    client = OpenAI(api_key="sk-6kG...")
    chunk_size = 300

    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(file_path, repo_path)

            if not should_ignore(relative_file_path, ignore_list):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    code = file.read().strip()

                    for chunk in chunk_code(code, chunk_size):
                        explanation = get_code_explanation(chunk, client, model)
                        write_to_file(output_file, relative_file_path, chunk, explanation)

# ----------------------------------- Main Function ----------------------------------- #
def main():
    """
    This code is the main function of a Python script that generates a markdown document.
    This document outlines the file structure and provides explanations for Python code
        in a given Git repository or directory.
    The script takes command line arguments to specify the path to the repository or
        directory, an optional preamble file, and an optional output file.
    If no command line arguments are provided, it prints a usage message and exits.
    The script then checks for an ignore file, reads it if it exists, and proceeds to
        create the output file.
    If a preamble file is specified, it writes the contents of the preamble file to the
        output file.
    Otherwise, it writes a default message to the output file.
    If the command line argument "-i" is provided, it creates the file structure in
        the output file.
    Finally, it processes the repository, applying the ignore list, and writes the results
        to the output file.

    Parameters:
        None

    Returns:
        None
    """
    if len(sys.argv) < 2:
        print("Usage: python gpt_project_inspector.py /path/to/git/repository_or_folder [-i] [-p /path/to/preamble.txt] [-o /path/to/output_file.txt]")
        sys.exit(1)

    path = sys.argv[1]
    ignore_file_path = os.path.join(path, ".gptignore")
    if sys.platform == "win32":
        ignore_file_path = ignore_file_path.replace("/", "\\")

    if not os.path.exists(ignore_file_path):
        here = os.path.dirname(os.path.abspath(__file__))
        ignore_file_path = os.path.join(here, ".gptignore")

    preamble_file = None
    if "-p" in sys.argv:
        preamble_file = sys.argv[sys.argv.index("-p") + 1]

    repository_or_folder = os.path.basename(os.path.normpath(path))
    output_file_path = f"{repository_or_folder}-output.md"
    if "-o" in sys.argv:
        output_file_path = sys.argv[sys.argv.index("-o") + 1]

    do_file_structure = False
    if "-i" in sys.argv:
        do_file_structure = True

    if os.path.exists(ignore_file_path):
        ignore_list = get_ignore_list(ignore_file_path)
    else:
        ignore_list = []

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        if preamble_file:
            with open(preamble_file, "r", encoding="utf-8") as pf:
                preamble_text = pf.read()
                output_file.write(f"## Preamble File:\n{preamble_text}\n")
        else:
            output_file.write("""
                The following text is a Git repository or a directory content inspector.
                It produces a simple markdown document outlining the file structure
                and providing explanations for python code in the repository or directory.\n
                """)

        if do_file_structure:
            create_file_structure(path, output_file_path)

        process_repository(path, ignore_list, output_file)

    with open(output_file_path, "a", encoding="utf-8") as output_file:
        output_file.write("---\n\n")

    print(f"Results were written to {output_file_path}.")


if __name__ == "__main__":
    main()
