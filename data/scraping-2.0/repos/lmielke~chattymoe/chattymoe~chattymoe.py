"""
##################### chattymoe MOE #####################
can do whatever you imagine


"""

import yaml, os, re, subprocess, time
from datetime import datetime as dt
import chattymoe.settings as sts
import chattymoe.helpers.general as hlp
from tabulate import tabulate as tb
import colorama as color

color.init()
_RED = color.Fore.RED
_YELLOW = color.Fore.YELLOW
_GREEN = color.Fore.GREEN
_CYAN = color.Fore.CYAN
_BLUE = color.Fore.BLUE
_WHITE = color.Fore.WHITE
_NONE = color.Style.RESET_ALL


import openai
openai.api_key = sts.apiKey

from chattymoe.prompt import Prompt

class Moe:
    """[summary]

    [description]
    """

    def __init__(self, *args, verbose: int = 0, **kwargs ) -> None:
        self.verbose = verbose
        self.running = False
        self.chat, self.comments = self.load_chat(*args, **kwargs)

    def post(self, *args, model, **kwargs):
        """
        Sends message to openai and handles the response.
        """
        r = openai.ChatCompletion.create(
            model=model, 
            messages=[{f: ct for f, ct in m.entry.items() if f in sts.msgFields} for m in self.chat]
            )
        self.chat.append(Prompt(r['choices'][-1]['message'], *args, **kwargs))
        self.print_message(*args, **kwargs)
        return r

    def print_message(self, *args, **kwargs):
        print(f"{_WHITE}assi:")
        assiMsg = {
                    'assi_in': hlp.insert_newline(self.chat[-2].entry.get('content')), 
                    'assi_out': [self.chat[-1].entry.get('content')], 
                    'lenChat': [len(self.chat)],
                    }
        assiMsg = tb(assiMsg, headers='keys', tablefmt='github')
        for line in assiMsg.split('\n'): print(f"\t{_YELLOW}{line}{_NONE}")
        time.sleep(5)


    def load_chat(self, *args, chat, **kwargs):
        """
        loads chats and comments
        chat: dict - chat history to prime the model with
        comments: dict - comments to add to the chat history and the chat going forward
        """
        fileName = f"{chat}.yml" if f"{chat}.yml" in os.listdir(sts.chatsDir) else sts.empty
        with open(os.path.join(sts.chatsDir, fileName), 'r') as f:
            params = yaml.safe_load(f)
        chat, comments = params['chat'], params.get('comments')
        if chat is None:
            self.running = True
        return chat, comments
    
    def confirm(self, *args, yes=False, **kwargs):
        if self.chat[-1].entry.get('content').strip().lower() in ['stop', 'stopp']:
            print(f"{_RED}User stopped the program{_NONE}")
            self.running = False
            return False
        if not yes:
            print(f"{_YELLOW}Moe.confirm: {_NONE}{self.chat[-1].entry = }")
            test = input("Test: ")
            userInput = input(f"Send/Stop? (y/stop): ").lower()
            exit()
            if userInput == 'y':
                return True
            elif userInput == 'stop':
                raise Exception("User stopped the program")
        return yes

    def check_exit_condition(self, *args, **kwargs):
        if cmd.lower() in ['stop', 'stopp']:
            raise Exception("User stopped the program")

    def initialize(self, *args, **kwargs):
        for i, prompt in enumerate(self.chat):
            self.chat[i] = Prompt(prompt, *args, **kwargs)
        self.running = True

    def check_correctness(self, *args, **kwargs):
        if len(self.lastAnswer) > sts.psMaxLen:
            return False, f"Answer too long! max: {sts.psMaxLen}"
        if '\n' in self.lastAnswer.strip():
            return False, f"Answer must be a single line!"
        return True, 'OK'

    def run(self, *args, **kwargs):
        # clean content for eve3ry chat entry
        while self.running:
            # prepare next prompt for futur yield
            if self.chat[-1].entry.get('role') not in ['user', 'system']:
                self.chat.append(Prompt(*args, **kwargs))
            if self.confirm(*args, **kwargs):
                self.post(*args, **kwargs)
            yield self


    def __str__(self, *args, **kwargs):
        dotPath = f"{__file__.replace(sts.projectDir, '').replace(os.sep, '.')}"
        return dotPath.strip('.')
