#!/usr/bin/env python3
# Author: Pierce Cohen
# Description: An interactive CLI for GPT models
# Version: 1.4.0

import openai
import os
import sys
from rich.console import Console
from rich.markdown import Markdown
import pyperclip
from playsound import playsound
import argparse
import json
import time
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter, Completer, Completion
from prompt_toolkit.document import Document
from rich.live import Live

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class CustomPathCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/save ") or text.startswith("/load "):
            command, partial_path = text.split(" ", 1)
            path_completer = PathCompleter()
            path_document = Document(partial_path, len(partial_path))
            for completion in path_completer.get_completions(path_document, complete_event):
                yield Completion(completion.text, completion.start_position,
                                 completion.display, completion.display_meta)

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

class ChatApplication:
    def __init__(self, system_message=None, model="gpt-3.5-turbo", clear_on_init=False):
        self.model = model
        self.system_message = system_message
        self.initialize_messages()
        self.console = Console()
        self.sound = True  # Set to False to disable sound
        if clear_on_init:
            clear_terminal()
    
    def initialize_messages(self, system_message=None, model=None):
        if system_message:
            self.system_message = system_message
        else:
            self.system_message = "You are a helpful assistant. Keep your answers concise when possible."
        if model:
            self.model = model
        self.messages = [{"role": "system", "content": self.system_message}]
    
    def add_message(self, role, content):
        message = {
            "role": role,
            "content": content
        }
        self.messages.append(message)
    
    def display_markdown(self, text):
        markdown = Markdown(text)
        self.console.print(markdown, end="")

    def try_chat_completion(self):
        response_content = ""
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True
            )
            for part in stream:
                content = part.choices[0].delta.content if part.choices[0].delta.content is not None else ''
                print(content, end='', flush=True)  # Print content as it's received
                response_content += content  # Accumulate the response content

                if 'finish_reason' in part.choices[0] and part.choices[0].finish_reason == 'stop':
                    print()
                    break  # Break the loop since the message is complete

        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return response_content.strip()
        
    def save_chat(self, filename):
        try:
            with open(filename, "w") as outfile:
                json.dump({"model": self.model, "messages": self.messages}, outfile, indent=4)
            print(f"Chat saved to '{filename}'.")
        except Exception as e:
            print(f"An error occurred while saving the chat: {e}")
    
    def load_chat(self, filename):
        try:
            with open(filename, "r") as infile:
                chat_data = json.load(infile)
            self.model = chat_data.get("model", self.model)
            self.messages = chat_data.get("messages", [])
            print(f"Chat loaded from '{filename}'.")
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred while loading the chat: {e}")
    
    def format_messages(self):
        formatted_messages = "\n".join(f"{message['role'].capitalize()}: {message['content']}"
                                       for message in self.messages)
        return formatted_messages

def main():
    parser = argparse.ArgumentParser(description="An interactive CLI for GPT models")
    parser.add_argument("-v", "--version", action="store_true", help="Show the version number and exit")
    parser.add_argument("--load", type=str, help="Load a chat from a file immediately upon launching the program")
    parser.add_argument("-q", "--query", type=str, help="Execute a query immediately upon launch")
    parser.add_argument('remainder', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.version:
        print("GPT-CLI version 1.4.0")
        sys.exit(0)

    chat_app = ChatApplication()

    if args.load:
        chat_app.load_chat(args.load)

    if args.query:
        chat_app.add_message("user", args.query)
        chat_app.try_chat_completion()
        print()
    else:
        print(f"Model: {chat_app.model}")

    session = PromptSession(completer=CustomPathCompleter())

    try:
        while True:
            user_input = session.prompt("You: ")

            if user_input.startswith('/'):
                command = user_input.split(' ')[0].lower()
                argument = user_input[len(command) + 1:].strip()

                if command == '/save':
                    chat_app.save_chat(argument)

                elif command == '/load':
                    chat_app.load_chat(argument)

                elif command == '/quit' or command == '/exit':
                    print("Exiting the GPT-CLI.")
                    break

                elif command == '/info':
                    user_messages_count = sum(1 for message in chat_app.messages if message['role'] == 'user')
                    assistant_messages_count = sum(1 for message in chat_app.messages if message['role'] == 'assistant')

                    table = Table(title="Chat Information", show_header=True, header_style="bold magenta")
                    table.add_column("Attribute", style="dim", width=20)
                    table.add_column("Value")
                    table.add_row("Model", chat_app.model)
                    table.add_row("System Message", chat_app.system_message)
                    table.add_row("User Messages", str(user_messages_count))
                    table.add_row("Assistant Messages", str(assistant_messages_count))
                    chat_app.console.print(table)

                elif command == '/clear':
                    clear_terminal()

                elif command == '/new':
                    chat_app.initialize_messages()
                    print("Started a new chat session.")

                elif command == '/system':
                    clear_terminal()
                    chat_app.initialize_messages(system_message=argument)
                    print("Started a new chat with a custom system message.")

                elif command == '/model':
                    chat_app.model = argument
                    if argument:
                        clear_terminal()
                        chat_app.initialize_messages()
                        print(f"Switched model to {argument} and started a new chat session.")
                    else:
                        print("Please specify a model name after the /model command.")
                elif command == '/copy':
                    pyperclip.copy(chat_app.messages[-1]['content'] if chat_app.messages else '')
                    print("Copied the last message to the clipboard.")

                elif command == '/paste':
                    pasted_content = pyperclip.paste()
                    chat_app.add_message("user", pasted_content)
                    chat_app.try_chat_completion()

                elif command == '/help':
                    print("Available commands:")
                    print("/paste - Paste clipboard content")
                    print("/copy - Copy the last response to the clipboard")
                    print("/new - Start a new chat")
                    print("/clear - Clear terminal window")
                    print("/system - Start a new chat with a custom system message")
                    print("/model - Start a new chat with the specified model")
                    print("/quit - Exit the program")
                    print("/info - Display info about the current chat session")
                    print("/save [FILENAME] - Save the chat to a file")
                    print("/load [FILENAME] - Load a chat from a file")
                    print("/help - Display this help message")

                else:
                    print(f"Unknown command: {command}")

            else:
                chat_app.add_message("user", user_input)
                assistant_response = chat_app.try_chat_completion()
                if assistant_response:
                    chat_app.add_message("assistant", assistant_response)
                print()

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()

