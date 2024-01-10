import argparse
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import openai
from dotenv import load_dotenv


def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def git_diff_output():
    process = subprocess.run(["git", "diff"], check=True, capture_output=True)
    return process.stdout.decode()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="chatgpt-commit", description="ChatGPT Commit Message Generator"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="set the temperature for the openai model",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_package_version('chatgpt_commit')}",
    )
    args = parser.parse_args()

    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    content = (
        "Based on the following Git diff output, could you help me to create an appropriate"
        " Git commit message in one line?"
        f"\n\n{git_diff_output()}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            temperature=args.temperature,
        )
    except openai.error.AuthenticationError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    for choice in response.choices:
        print(choice.message.content)


if __name__ == "__main__":
    main()
