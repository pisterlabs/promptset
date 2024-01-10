#!/usr/bin/env python3
# WORKERS OF THE WORLD UNITE ✊
from typing import Dict
import re
import readline
import os
import sys
import subprocess
import venv


def setup_virtual_environment():
    venv_path = "venv"
    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)

    if sys.platform == "win32":
        activate_script = os.path.join(venv_path, "Scripts")
    else:
        activate_script = os.path.join(venv_path, "bin")

    os.environ["PATH"] = f"{activate_script}{os.pathsep}{os.environ['PATH']}"

    try:
        global Text, Console, Panel, box
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "rich", "openai"]
        )
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text


setup_virtual_environment()
import openai  # noqa: E402
from rich.table import Table  # noqa: E402

console = Console()


class ChatHandler:
    def __init__(self) -> None:
        self.set_api_key()
        self.conversation = []

    def set_api_key(self):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            print("Error: Could not set API key")
            sys.exit(1)

    def chat_gpt(self, conversation):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=conversation, temperature=0.1
            )
            content = response["choices"][0]["message"]["content"]
            return {"role": "assistant", "content": content}
        except Exception as e:
            console.print(e, style="red")
            sys.exit(1)

    def update_conversation(self, append: Dict = None) -> None:
        if append is not None:
            self.conversation.append(append)
            role = append["role"]
            content = append["content"]
            console.print(f"{role.capitalize()}: {content}", style="green")

    def extract_code_block(self, response: str) -> str:
        code_block_pattern = r"```(.+?)```"
        code_snippet_pattern = r"`(.+?)`"
        code_block_match = re.search(code_block_pattern, response, re.S)
        code_snippet_match = re.search(code_snippet_pattern, response, re.S)
        if code_block_match:
            return code_block_match.group(1).strip()
        elif code_snippet_match:
            return code_snippet_match.group(1).strip()
        else:
            save_flag = input(
                "would you like to save this reply as the command? (y/n): "
            )
            if save_flag.lower() == "y":
                return response.strip()
            else:
                pass

    def request_explanation(self, command: str) -> str:
        explanation_request = (
            f"give a concise explanation of the following bash command: {command}"
        )
        self.update_conversation({"role": "user", "content": explanation_request})
        explanation = self.chat_gpt(self.conversation)["content"]
        self.update_conversation({"role": "assistant", "content": explanation})

    def refine_prompt(self, user_feedback: str) -> str:
        refined_prompt = self.chat_gpt(self.conversation)["content"]
        self.update_conversation({"role": "assistant", "content": refined_prompt})
        return refined_prompt

    def verify_command(self, command: str) -> str:
        self.update_conversation(
            {
                "role": "user",
                "content": f"echo the command if correct, or revise if there are errors: {command}",
            }
        )
        corrected_command = self.chat_gpt(self.conversation)["content"]
        edited_command = get_prompt_input(corrected_command)
        self.update_conversation({"role": "user", "content": edited_command})
        return edited_command.strip()

    def print_conversation(self):
        conversation_text = Text("\n\nConversation so far:\n\n", style="bold")
        for message in self.conversation:
            role = message["role"]
            content = message["content"]
            message_style = "green" if role == "assistant" else "white"
            conversation_text.append(f"{role.capitalize()}: ", style=message_style)
            conversation_text.append(f"{content}\n", style=message_style)
        panel = Panel(conversation_text, box=box.ROUNDED, style="white on black")
        console.print(panel)


def get_prompt_input(prompt: str) -> str:
    readline.set_startup_hook(lambda: readline.insert_text(prompt).replace("\\", ""))
    try:
        user_input = input("Write a command (q for quit): ")
        if user_input == "q":
            sys.exit(0)
        return user_input
    finally:
        readline.set_startup_hook()


def welcome_to_chatbash() -> None:
    title = Text("Welcome to chatbash", style="bold_underline")
    description = Text(
        "This program is designed to help you collaborate with chatGPT to craft a bash command."
    )

    instructions = Table(box=None, expand=True)
    instructions.add_column("Instructions")
    instructions.add_row(
        "Use the feedback option to refine your prompt, or ask for a new one entirely!"
    )
    instructions.add_row(
        "Tip: request for your original bash command if the program overwrote it. It would be happy to oblige."
    )
    console.print(Panel(title, expand=True))
    console.print(description)
    console.print(instructions)


def main():
    welcome_to_chatbash()
    chat = ChatHandler()
    args = sys.argv[1:]
    quick_explain = False

    if "-x" in args:
        quick_explain = True
        args.remove("-x")

    prompt = " ".join(args)

    if prompt == "":
        if quick_explain:
            print("Error: No command provided for quick explanation.")
            sys.exit(1)
        else:
            prompt = input("Write a natural language command, or 'q' to quit: ")
            if prompt == "q":
                sys.exit(0)

    if quick_explain:
        explanation = chat.request_explanation(prompt)
        print("")
        sys.exit(0)

    conversation = [
        {
            "role": "system",
            "content": "Your goal is to collaborate with the user to generate a bash command. Only respond with one bash command per reply",
        },
        {
            "role": "user",
            "content": f"given the following prompt, generate a bash command. do not use any formatting. Do not provide any commentary: {prompt}",
        },
    ]

    chat_gpt_response = chat.chat_gpt(conversation)["content"]
    chat.update_conversation({"role": "assistant", "content": chat_gpt_response})
    command = chat.extract_code_block(chat_gpt_response)

    while True:
        console.print(
            "Careful! Bash commands are powerful... make sure you understand the prompt",
            style="red",
        )
        console.print(f"Command: {command}", style="bold")
        run_flag = input(
            "1. run? [(r)un/(q)uit/e(x)plain/(f)eedback/(e)dit/(p)rint conversation/(t)ry again]: "
        )
        match run_flag:
            case "r":
                try:
                    subprocess.run(command, shell=True, check=True)
                    sys.exit(0)
                except subprocess.CalledProcessError as e:
                    print(f"Error executing the command: {e}")
                    sys.exit(1)

            case "x":
                print("\n\n")
                chat.request_explanation(command)
            case "f":
                user_feedback = input("Feedback: ")
                print("\n\n")
                chat.update_conversation({"role": "user", "content": user_feedback})
                refined_prompt = chat.refine_prompt(user_feedback)
                command = chat.extract_code_block(refined_prompt)
            case "e":
                print("\n\n")
                corrected_command = chat.verify_command(command)
                command = chat.extract_code_block(corrected_command)
                console.print(command, style="blue")
            case "p":
                print("\n\n")
                chat.print_conversation()
            case "q":
                sys.exit(0)
            case "t":
                print("\n\n")
                if chat.conversation:
                    chat.conversation.pop()
                chat_gpt_response = chat.chat_gpt(chat.conversation)["content"]
                chat.update_conversation(
                    {"role": "assistant", "content": chat_gpt_response}
                )
                command = chat.extract_code_block(chat_gpt_response)
            case _:
                continue


if __name__ == "__main__":
    main()
