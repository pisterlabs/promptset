#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import openai
from termcolor import colored
import platform
import distro
import tiktoken
from prompt_toolkit import ANSI, PromptSession, prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory


class OpenAIHelper:
    """A class that handles the OpenAI API calls."""

    def __init__(self, model_name="gpt-3.5-turbo", max_tokens=4096):
        """Initializes the OpenAIHelper class."""
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if self.api_key == "":
            print(colored("Error: OPENAI_API_KEY is not set", "red"), file=sys.stderr)
            exit(1)
        openai.api_key = self.api_key

        self.max_tokens = max_tokens
        self.remaning_tokens = max_tokens
        self.model_name = model_name
        self.all_messages = []
        self.get_model_for_encoding(model_name)
        self.os_name, self.shell_name = OSHelper.get_os_and_shell_info()

        self.functions = [
            {
                "name": "get_commands",
                "description": f"Get a list of {self.shell_name} commands on an {self.os_name} machine",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "A valid command string"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the command"
                                    }
                                },
                                "required": ["command"]
                            },
                            "description": "List of terminal command objects to be executed"
                        },
                        "response": {
                            "type": "string",
                            "description": "Give me a detailed description of what you want to do",
                        }
                    },
                    "required": ["commands", "response"]
                }
            }
        ]

    def get_model_for_encoding(self, model: str):
        """Returns the number of tokens used by a list of messages."""
        if "gpt-3.5-turbo" in model:
            self.tokens_per_message = 4
            self.tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-4" in model:
            self.tokens_per_message = 3
            self.tokens_per_name = 1
        else:
            print(colored(
                f"get_model_for_encoding() is not implemented for model {model}.", "red"), file=sys.stderr)
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(
                colored("Warning: model not found. Using cl100k_base encoding.", "yellow"))
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def get_all_message_tokens(self):
        """Returns the number of tokens used by a list of messages."""
        num_tokens = 0
        for message in self.all_messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                if key == "name":
                    num_tokens += self.tokens_per_name
                if value is None:
                    continue
                if type(value) == dict:
                    for k, v in value.items():
                        num_tokens += len(self.encoding.encode(v))
                    continue
                num_tokens += len(self.encoding.encode(value))

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def truncate_outputs(self, outputs):
        """Truncates the outputs list so that the total tokens fit the max_tokens limit."""
        max_tokens = self.max_tokens - self.max_tokens // 2
        outputs_tokens = []
        total_tokens = 0
        for i in range(len(outputs)):
            tokens = {"index": i,
                      "command": self.encoding.encode(outputs[i]["command"]),
                      "stdout": self.encoding.encode(outputs[i]["stdout"]),
                      "stderr": self.encoding.encode(outputs[i]["stderr"])}

            tokens["total"] = len(tokens["command"]) + \
                len(tokens["stdout"]) + len(tokens["stderr"])
            total_tokens += tokens["total"]

            outputs_tokens.append(tokens)

        if total_tokens <= max_tokens:
            return outputs
        tokens_to_remove = total_tokens - max_tokens

        # Sort by the length of tokens in ascending order
        outputs_tokens.sort(key=lambda x: x["total"])

        half_max_tokens = max_tokens // 2
        total_tokens_half = 0
        index_to_start_truncate = 0

        for tokens in outputs_tokens:
            if total_tokens_half + tokens["total"] > half_max_tokens:
                break
            total_tokens_half += tokens["total"]
            index_to_start_truncate += 1

        remaining_tokens = total_tokens - total_tokens_half

        for i in range(index_to_start_truncate, len(outputs_tokens)):
            # count procent of remaining tokens
            procent = outputs_tokens[i]["total"] / remaining_tokens
            tokens_to_remove_in_this_iteration = int(
                tokens_to_remove * procent)
            # print(colored(
            #     f"procent: {procent} tokens: {outputs_tokens[i]['total']} remaining_tokens: {remaining_tokens} tokens_to_remove: {tokens_to_remove} tokens_to_remove_in_this_iteration: {tokens_to_remove_in_this_iteration}", "green"))

            if tokens_to_remove_in_this_iteration > 0:
                tokens_to_remove -= tokens_to_remove_in_this_iteration
                total_tokens -= tokens_to_remove_in_this_iteration
                outputs_tokens[i]["stdout"] = outputs_tokens[i]["stdout"][:-
                                                                          tokens_to_remove_in_this_iteration]

                outputs_tokens[i]["total"] = len(outputs_tokens[i]["command"]) + \
                    len(outputs_tokens[i]["stdout"]) + \
                    len(outputs_tokens[i]["stderr"])

                outputs[outputs_tokens[i]["index"]]["stdout"] = self.encoding.decode(
                    outputs_tokens[i]["stdout"])

        # print(colored(
        #     f"total_tokens: {total_tokens} max_tokens: {max_tokens} tokens_to_remove: {tokens_to_remove}", "green"))
        return outputs

    def truncate_chat_message(self):
        """Truncates the chat message list so that the total tokens fit the max_tokens limit."""
        max_tokens = self.max_tokens - 400
        all_message_tokens = self.get_all_message_tokens()
        # print(colored(
        #     f"on start truncate_chat_message() all message tokens: {all_message_tokens}", "green"))

        while all_message_tokens > max_tokens:
            try:
                message = self.all_messages.pop(0)
                all_message_tokens = self.get_all_message_tokens()
                # print(colored(
                #     f"all message tokens: {all_message_tokens} message: {message}", "green"))
            except IndexError:
                break
            pass

    def get_commands(self, prompt):
        """Returns a list of commands to be executed."""
        message = {
            "role": "user",
            "content": prompt}
        self.all_messages.append(message)

        self.truncate_chat_message()

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.all_messages,
            functions=self.functions,
            function_call={"name": "get_commands"},
        )

        commands = None
        response_description = None
        if response is not None:
            if isinstance(response, dict):
                self.remaning_tokens = self.max_tokens - \
                    response['usage']['total_tokens']

                response_message = response['choices'][0].message
                if response_message.get("function_call"):
                    function_name = response_message["function_call"]["name"]

                    if function_name == "get_commands":
                        message_to_add = response_message.to_dict()
                        message_to_add["function_call"] = response_message["function_call"].to_dict(
                        )
                        self.all_messages.append(message_to_add)
                        arguments = json.loads(
                            response_message["function_call"]["arguments"])
                        commands = arguments["commands"]
                        if "response" in arguments:
                            response_description = arguments["response"]
                    else:
                        print(colored(
                            f"Warning: function {function_name} is not implemented.", "yellow"))

        return commands, response_description

    def send_commands_outputs(self, outputs):
        """Returns a list of commands to be executed."""
        outputs = self.truncate_outputs(outputs)

        outputs = json.dumps(outputs)
        message = {
            "role": "function",
            "name": "get_commands",
            "content": outputs}
        self.all_messages.append(message)

        prompt = "Explain the result in detail."
        message = {
            "role": "user",
            "content": prompt}
        self.all_messages.append(message)

        self.truncate_chat_message()

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.all_messages,
            functions=self.functions,
            function_call='auto',
        )

        response_content = None
        commands = None
        if response is not None:
            if isinstance(response, dict):
                self.remaning_tokens = self.max_tokens - \
                    response['usage']['total_tokens']

                response_message = response['choices'][0].message
                response_message = response_message.to_dict()
                self.all_messages.append(response_message)

                response_content = response_message['content']

                if response_message.get("function_call"):
                    function_name = response_message["function_call"]["name"]

                    if function_name == "get_commands":
                        message_to_add = response_message
                        message_to_add["function_call"] = response_message["function_call"].to_dict(
                        )
                        self.all_messages.append(message_to_add)
                        arguments = json.loads(
                            response_message["function_call"]["arguments"])
                        commands = arguments["commands"]
                    else:
                        print(colored(
                            f"Warning: function {function_name} is not implemented.", "yellow"))

        return response_content, commands


class CommandHelper:
    """Helper class for executing commands."""

    @staticmethod
    def run_shell_command(command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, universal_newlines=True)
        stdout = []

        # print(f"Output")
        for line in process.stdout:
            print(line, end="")
            stdout.append(line)

        stderr_data = process.stderr.read()
        if stderr_data:
            print(colored(f"Error\n{stderr_data}", "red"))

        process.wait()

        output = {"command": command, "stdout": ''.join(
            stdout), "stderr": stderr_data}

        if output["stdout"] != "":
            stdout_lines = output["stdout"].split("\n")
            for i, line in enumerate(stdout_lines):
                if "KEY" in line:
                    post_key_content = line[line.find("KEY") + 3:].strip()
                    if "=" in post_key_content:
                        stdout_lines[i] = line.split("=")[0] + "=<API_KEY>"
            output["stdout"] = "\n".join(stdout_lines)
        return output


class OSHelper:
    """Helper class for getting OS and shell information."""

    @staticmethod
    def get_os_and_shell_info():
        """Returns the OS and shell information."""
        os_name = platform.system()
        shell_name = os.path.basename(os.environ.get("SHELL", ""))
        if os_name == "Linux":
            os_name += f" {distro.name()}"
        elif os_name == "Darwin":
            os_name += f" {platform.mac_ver()[0]}"
        elif os_name == "Windows":
            os_name += f" {platform.release()}"
        return os_name, shell_name


class Application:
    """Main application class."""

    def __init__(self, openai_helper: OpenAIHelper, command_helper: CommandHelper):
        """Initializes the application."""
        self.openai_helper = openai_helper
        self.command_helper = command_helper
        self.session = PromptSession(history=FileHistory(os.path.expanduser(
            '~') + "/.gpts_history"), auto_suggest=AutoSuggestFromHistory())

    def interpret_and_execute_command(self, user_prompt):
        """Interprets and executes the command."""
        if user_prompt == "e":
            self.manual_command_mode()
        else:
            self.auto_command_mode(user_prompt)

    def manual_command_mode(self):
        """Manual command mode."""
        print(colored("Manual command mode activated. Please enter your command:", "green"))
        command_str = self.session.prompt("")
        command_output = self.command_helper.run_shell_command(command_str)
        outputs = [command_output]

        response, commands = self.openai_helper.send_commands_outputs(outputs)
        print(colored(f"{response}", "magenta"))

        if commands is not None:
            self.execute_commands(commands)

    def auto_command_mode(self, user_prompt):
        """Auto command mode."""
        commands, response_description = self.openai_helper.get_commands(
            user_prompt)
        if response_description is not None:
            print(colored(f"{response_description}\n", "magenta"))
        if commands is not None:
            self.execute_commands(commands)
        else:
            print(colored("No commands found", "red"))

    def execute_commands(self, commands):
        """Executes the commands."""
        outputs = []
        action = ""
        while commands is not None:
            commands_list = [command["command"] for command in commands]
            print(colored(f"List of commands {commands_list}", "magenta"))
            for command in commands:
                command_str = command["command"]
                print(colored(f"{command['description']}", "magenta"))
                print(colored(f"{command_str}", "blue"))

                if action.lower() != "a":
                    action = prompt(ANSI(
                        colored(f"Do you want to run (y), edit (e), or execute all (a) commands? (y/e/a/N): ", "green")))

                if action.lower() == "e":
                    command_str = self.session.prompt(ANSI(
                        colored("Enter the modified command: ", "cyan")), default=command_str)
                    action = prompt(ANSI(
                        colored(f"Do you want to run the command? (y/N): ", "green")))

                if action.lower() in ["y", "a"]:
                    output = self.command_helper.run_shell_command(command_str)
                    outputs.append(output)
                else:
                    print(colored("Skipping command", "yellow"))
                    # break

            if len(outputs) > 0:
                response, commands = self.openai_helper.send_commands_outputs(
                    outputs)
                print(colored(f"{response}", "magenta"))
                outputs = []
                action = ""
            else:
                commands = None

    def run(self):
        """Runs the application."""
        os_name, shell_name = OSHelper.get_os_and_shell_info()
        print(
            colored(f"Your current environment: Shell={shell_name}, OS={os_name}", "green"))
        print(colored(
            "Type 'e' to enter manual command mode or 'q' to quit, (tokens left)\n", "green"))

        while True:
            try:
                user_input = self.session.prompt(
                    ANSI(colored(f"ChatGPT ({self.openai_helper.remaning_tokens}): ", "green")))
                if user_input.lower() == 'q':
                    break
                self.interpret_and_execute_command(user_input)
            except subprocess.CalledProcessError as e:
                print(colored(
                    f"Error: Command failed with exit code {e.returncode}: {e.output}", "red"), file=sys.stderr)
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                print(
                    colored(f"Error of type {type(e).__name__}: {e}", "red"), file=sys.stderr)
                print(colored("Exiting...", "yellow"))


if __name__ == "__main__":
    """Main entry point."""
    openai_helper = OpenAIHelper(  # model_name="gpt-3.5-turbo"
        model_name="gpt-3.5-turbo-16k-0613",
        max_tokens=16 * 1024
    )
    command_helper = CommandHelper()
    application = Application(openai_helper, command_helper)
    application.run()
