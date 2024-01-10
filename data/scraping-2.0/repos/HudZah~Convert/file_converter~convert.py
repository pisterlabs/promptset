import argparse
import subprocess
from openai import OpenAI
import tempfile
import os


class CommandParser:
    def __init__(self, query, history_file_path):
        self.query = query
        self.history_file_path = history_file_path

    def parse(self):
        client = OpenAI()

        # Retrieve the history file
        with open(self.history_file_path, "r") as f:
            history = f.read()

        history_prompt = (
            "For context, here are recent question and answers, so if the current question is ambigous see if theres context here.\n\n"
            + history
        )

        system_prompt = f"""
        You are a command line utility that quickly and succinctly converts images and videos and manipulates them. When a user asks a question, you respond with the most relevant command that can be executed within the command line, along with the required packages that need to be installed. If the command has pre-requisite tools to install, install them first before proceeding. Your responses should be clear and console-friendly, remember the command you output must be directly copyable and would execute in the command line.

Here's how your responses should look:

EXAMPLE 1

<Users Question>
conv file.webp to png
<Your Answer>
`'dwebp file.webp -o file.png'`

EXAMPLE 2

<Users Question>
rotate an image by 90 degrees
<Your Answer>
`brew install imagemagick`
`convert file.png -rotate 90 rotated_file.png`

EXAMPLE 3

<Users Question>
convert a video in /path/to/video.mp4 to a gif
<Your Answer>
`ffmpeg -i /path/to/video.mp4 /path/to/video.gif`

EXAMPLE 4

<Users Question>
avif to png for file.avif
<Your Answer>
`magick file.avif file.png`

"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Answer this as briefly as possible: " + self.query,
            },
        ]

        if history:
            messages.insert(
                1,
                {
                    "role": "user",
                    "content": "For context, here are recent question and answers, so if the current question is ambigous see if theres context here. Use this to also keep file locations in mind, in case files are moved around or names changed, use the latest context from here.\n\n"
                    + history_prompt,
                },
            )

        completion_stream = client.chat.completions.create(
            messages=messages,
            model="gpt-4-1106-preview",
            stream=True,
            max_tokens=100,
        )

        response = ""

        for chunk in completion_stream:
            response += chunk.choices[0].delta.content or ""

        # Write the last 5 commands to the history file
        with open(self.history_file_path, "a") as f:
            f.write(f"Question: {self.query}\nAnswer: {response}\n\n")

        return response


class CommandExecutor:
    @staticmethod
    def execute(command):
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"\033[1;34;40mExecuted: {command}\033[0m")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")


def clear_history(history_file_path):
    with open(history_file_path, "w") as f:
        f.write("")


def main():
    temp_dir = tempfile.gettempdir()
    history_file_path = os.path.join(temp_dir, "history.txt")
    if not os.path.exists(history_file_path):
        with open(history_file_path, "w") as f:
            pass

    parser = argparse.ArgumentParser(
        description="Conv is a command line tool to easily execute file conversions, image manipulations, and file operations quickly."
    )
    parser.add_argument("query", type=str, nargs="*", help="The query to be processed.")
    parser.add_argument("--clear", action="store_true", help="Clear the history.")

    args = parser.parse_args()

    if args.clear:
        clear_history(history_file_path)
        print("History cleared.")
        return

    if args.query is None:
        print("Usage: python script.py 'conv <query>' or '--clear' to clear history")
        return

    query = " ".join(args.query)
    print(query)

    command_parser = CommandParser(query, history_file_path)
    system_command = command_parser.parse()

    if system_command:
        CommandExecutor.execute(system_command)
    else:
        print("Could not parse or execute the command.")


if __name__ == "__main__":
    main()
