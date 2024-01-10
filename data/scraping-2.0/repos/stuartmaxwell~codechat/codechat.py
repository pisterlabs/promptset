import argparse
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

from openaichat import chat_with_openai


def read_file_content(file_paths: list[str]) -> str:
    """
    Read the content of multiple files and concatenate them into a single string.

    Args:
        file_paths (List[str]): A list of file paths.

    Returns:
        str: The concatenated content of the files, with each file separated by a
        header.
    """
    content = ""
    for file_path in file_paths:
        with open(file_path, "r") as file:
            content += f"\n--- {file_path} ---\n"
            content += file.read()
    return content


def print_syntax(file_path: str) -> None:
    """
    Prints the contents of a file at the given path with syntax highlighting.

    Args:
        file_path: The path of the file to print.
    """
    console = Console()
    syntax = Syntax.from_path(file_path)
    console.print(syntax)


def print_markdown(contents: str) -> None:
    """
    Prints the given markdown `contents` in a formatted way using the Rich library.

    Args:
        contents: The markdown content to be printed.
    """
    console = Console()
    md = Markdown(contents, inline_code_theme="monokai")
    console.print(md)


def get_user_input(code: Optional[str] = None) -> Optional[str]:
    """
    Prompts the user for a question and returns it.

    Args:
        code (Optional[str]): The contents of the file, if already read.

    Returns:
        The user input as a string, or None if the user types 'exit'.
    """
    if code:
        question = input("Ask a question (type 'exit' to quit): ")
        response = f"{question}\n\n{code}"
    else:
        question = input("Ask another question (type 'exit' to quit): ")
        response = f"{question}"

    return None if question.lower() == "exit" else response


def main() -> None:
    """
    Main function to run the script. It uses argparse to allow users to enter multiple
    files at the command line, reads the content of those files, and then calls the
    chat_with_openai() function with the user's question and the code content as
    context.
    """
    parser = argparse.ArgumentParser(
        description="Ask questions about code from multiple files."
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="List of files to include in the context",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompt",
        help="Custom initial prompt for the conversation",
    )

    args = parser.parse_args()

    code_content = read_file_content(args.files)
    system_content = (
        args.prompt or "You are a helpful assistant with expertise in programming."
    )

    while True:
        user_input = get_user_input(code_content)
        code_content = None

        if not user_input:
            break

        try:
            code_info = chat_with_openai(system_content, user_input)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        if code_info:
            print_markdown(code_info)


if __name__ == "__main__":
    main()
