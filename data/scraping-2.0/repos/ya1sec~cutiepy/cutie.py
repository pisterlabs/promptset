import os
import sys
import openai
import platform
import subprocess
import distro

from termcolor import colored
from colorama import init
from dotenv import load_dotenv

art = """
                    _____
                ___/     \___
               `-._)     (_,-`
                   \O _ O/
                    \ - /
                     `-(
                      ||
                     _||_
                    |-..-|
                    |/. \|
                    |\__/|
                  ._|//\\|_,
                  `-((  ))-'
                   __\\//__ gnv
                   >_ /\ _<,
                     '  '
"""

print(colored(art, "yellow"))

cur_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

SAFETY_SWITCH = True


class CutiePy:
    def __init__(self, arguments):
        self.safety_switch = True

        self.model = "gpt-3.5-turbo"
        self.openai_api_key = OPENAI_API_KEY

        self.shell = self.get_shell()
        self.os = self.get_os()

        self.user_prompt = self.get_user_prompt(arguments)

        full_prompt = self.get_full_prompt(self.user_prompt)
        self.system_prompt = full_prompt[1]
        self.prompt = full_prompt
        # self.response = self.get_response()
        # self.reply = self.get_reply()

    def get_shell(self):
        shell = os.environ.get("SHELL")
        if shell is None:
            shell = "unknown"
        return shell

    def get_os(self):
        os_name = platform.system()
        if os_name == "Linux":
            return "Linux/" + distro.name(pretty=True)
        elif os_name == "Windows":
            return os_name
        elif os_name == "Darwin":
            return "Darwin/macOS"

    def get_user_prompt(self, arguments):
        user_prompt = " ".join(arguments)
        # return " ".join(sys.argv[1:])
        return user_prompt

    # Construct the prompt
    def get_full_prompt(self, user_prompt, explain=False):
        ## Find the executing directory (e.g. in case an alias is set)
        ## So we can find the prompt.txt file
        yolo_path = os.path.abspath(__file__)
        prompt_path = os.path.dirname(yolo_path)

        ## Load the prompt and prep it
        if explain == True:
            # If explain is true, user prompt will be the command
            prompt_file = os.path.join(prompt_path, "prompts/explain.txt")
        else:
            prompt_file = os.path.join(prompt_path, "prompts/prompt.txt")

        pre_prompt = open(prompt_file, "r").read()
        pre_prompt = pre_prompt.replace("{shell}", self.shell)
        pre_prompt = pre_prompt.replace("{os}", self.os)
        prompt = pre_prompt + user_prompt

        # be nice and make it a question
        if prompt[-1:] != "?" and prompt[-1:] != ".":
            prompt += "?"

        return prompt

    def get_response(self):
        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        return response

    def get_cmd(self):
        response = self.get_response()
        # reply = response["choices"][0]["message"]["content"]
        cmd = response.choices[0].message.content.strip()
        return cmd

    def get_explanation(self, cmd):
        pass

    def run(self):
        cmd = self.get_cmd()
        # number of characters in the command
        cmd_len = len(cmd)

        # print(colored("Are you sure you want to run this command?", "red"))
        print("Are you sure you want to run this command?")
        print(colored("Y: yes", "yellow"))
        print(colored("E: explain the command", "yellow"))
        print(colored("N: cancel", "yellow"))

        print("\n")
        print(colored("-" * cmd_len, "red"))
        print(colored(cmd, "green"))
        print(colored("-" * cmd_len, "red"))
        print("\n")

        # print(colored("Type 'y' to continue, 'e' to explain, or anything else to cancel", "red"))

        answer = input()
        if answer == "y":
            # print(colored("Running command...", "green"))
            print(colored(cmd, "green"))
            if self.shell == "powershell.exe":
                subprocess.run([self.shell, "/c", cmd], shell=False)
            else:
                # Unix: /bin/bash /bin/zsh: uses -c both Ubuntu and macOS should work, others might not
                subprocess.run([self.shell, "-c", cmd], shell=False)


if __name__ == "__main__":
    # Parse arguments and make sure we have at least a single word
    if len(sys.argv) < 2:
        sys.exit(-1)

    # Get the user prompt
    arguments = sys.argv[1:]

    # Create the CutiePy object
    cutie = CutiePy(arguments).run()




###############################################################################

# Borrowing from Yolo - (Copyright (c) 2023 wunderwuzzi23 MIT License)
# https://github.com/wunderwuzzi23/yolo-ai-cmdbot

"""

MIT License

Copyright (c) 2023 wunderwuzzi23
Greetings from Seattle! 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""