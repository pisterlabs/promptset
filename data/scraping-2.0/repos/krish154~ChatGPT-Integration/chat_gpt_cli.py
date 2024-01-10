#!/usr/bin/env python
# Chat GPT integration with commnd line
import os
import openai
import subprocess
import signal
import readline
import shutil
import pty
import sys

openai.api_key = "YOUR_OPENAI_API_HERE"

history_file_path = "history.txt"

# Color codes
color_codes = {
    'green': '\033[32m',
    'white': '\033[37m',
    'blue': '\033[34m',
    'reset': '\033[0m',
    'red': '\033[31m'
}


def colorize(text, color):
    return f"{color_codes[color]}{text}{color_codes['reset']}"


def chat_with_gpt(response, length):
    gpt_prompt = response

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=gpt_prompt,
        temperature=0.5,
        max_tokens=length,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response['choices'][0]['text']


def save_history(history):
    with open(history_file_path, "a") as f:
        f.write(history + "\n")


def load_history():
    if not os.path.exists(history_file_path):
        # Create the history file if it doesn't exist
        with open(history_file_path, "w") as f:
            pass

    with open(history_file_path, "r") as f:
        lines = f.readlines()
        return [line.strip() for line in lines]


def execute_command(command):
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True  # Enable line-based output
        )

        output_lines = []
        for line in process.stdout:
            # Colorize directory names in blue
            if os.path.isdir(line.strip()):
                output_lines.append(colorize(line.strip(), 'blue'))
            # Colorize executable files in green
            elif os.access(line.strip(), os.X_OK):
                output_lines.append(colorize(line.strip(), 'green'))
            else:
                output_lines.append(line.strip())  # Append each line of output

        num_files = len(output_lines)
        num_columns = min(3, num_files)

        # Get the current terminal width
        terminal_width, _ = shutil.get_terminal_size((80, 20))

        # Calculate the column width based on the available terminal width
        column_width = min(terminal_width // num_columns, 50)

        row_width = num_columns * column_width

        for i in range(0, num_files, num_columns):
            row_files = output_lines[i:i + num_columns]
            row_files_padded = [file.ljust(column_width) for file in row_files]
            row = ''.join(row_files_padded)
            print(row)

        for line in process.stderr:
            print(line.strip())  # Print each line of error (if any)

        process.wait()  # Wait for the process to finish before continuing
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Error executing command: {e}")

def is_command_exists(command):
    return any(
        os.access(os.path.join(path, command), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )


def handle_interrupt(signal, frame):
    print("\nProgram interrupted. Resuming...")


def read_input(prompt):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline().rstrip('\n')
    return line


if __name__ == '__main__':
    history = load_history()

    signal.signal(signal.SIGINT, handle_interrupt)

    while True:
        try:
            current_working_directory = os.getcwd()
            prompt = f"{colorize('┌──(', 'blue')}{colorize('ChatGPT💀shell', 'red')}{colorize(')-[', 'blue')}{colorize(current_working_directory, 'white')}{colorize(']', 'blue')}\n{colorize('└─>> ', 'blue')}"

            search = read_input(colorize(prompt, 'blue'))

            if not search:
                continue

            if search.lower() in ['quit', 'exit']:
                break

            if search == "\033[A":
                # Upper arrow key - Retrieve previous command from history
                previous_command = readline.get_history_item(readline.get_current_history_length() - 1)
                if previous_command:
                    search = previous_command
                    print(search)

            if search == "\033[B":
                # Lower arrow key - Retrieve next command from history
                next_command = readline.get_history_item(readline.get_current_history_length() + 1)
                if next_command:
                    search = next_command
                    print(search)

            if search.startswith("cd"):
                # Change directory
                directory = search.split("cd", 1)[-1].strip()
                if directory:
                    try:
                        os.chdir(directory)
                    except FileNotFoundError:
                        print(f"Directory not found: {directory}")
                    except NotADirectoryError:
                        print(f"Not a directory: {directory}")
                    except PermissionError:
                        print(f"Permission denied: {directory}")
                    except Exception as e:
                        print(f"Error changing directory: {e}")
                    continue

            if is_command_exists(search.split()[0]):
                # Command execution
                command = search
                execute_command(command)
            else:
                # Chatbot response
                history.append(search)
                response = "\n".join(history[-3:])
                chat_response = chat_with_gpt(response, 500)

                save_history(f"{response}\n{chat_response}")

                print(chat_response)
        except KeyboardInterrupt:
            print("\nProgram interrupted. Resuming...")
        except Exception as e:
            print(f"Error: {e}")
