from typing import Type
import openai
from src.inputparser import InputParser
from src.action import Action
from src.gpt import GPT
from src.client import Client
from src.conversation import Conversation
from src.systemprompt import SystemPrompt
from src.logger import Logger
import subprocess
import ast
import sys
import os
from src.cmdshelp import HELP_STRING


class Dispatcher:
    """Dispatches functions and manages conversation state."""
    def __init__(self):
        self.thinking: bool = False

    def dispatch(self, action: Action):
        """Turns an Action into a function"""
        if action.command == 'stop':
            self.thinking = True  # >stop
            return self.silence
        elif action.command == 'start':
            self.thinking = False  # >start
            return self.silence
        elif self.thinking and action.command == 'chat':
            return self.think
        elif action.command == 'chat':
            return self.speak
        elif action.command == 'exec':
            return self.execute
        elif action.command == 'insert':
            return self.insert
        elif action.command == 'show':
            return self.show
        elif action.command == 'flush':
            return self.flush
        elif action.command == 'save':
            return self.save
        elif action.command == 'set':
            return self.set
        elif action.command == 'reset':
            return self.reset
        elif action.command == 'help':
            return self.help
        elif action.command == 'exit':
            return self.goodbye
        else:
            return self.silence

    @staticmethod
    def silence(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Whereof one cannot speak, thereof one must be silent"""
        return gpt, conversation, logger

    @staticmethod
    def think(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """>stop While thinking: Append user input to conversation.query."""
        conversation = conversation.think(action.raw_input)
        return gpt, conversation, logger

    @staticmethod
    def speak(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Send query to GPT API and receive response"""
        try:
            if action.raw_input == '':
                conversation = conversation.speak(content=conversation.query)
            elif action.raw_input != '' and conversation.query == '':
                conversation = conversation.speak(content=action.raw_input)
            elif action.raw_input != '' and conversation.query != '':
                conversation = conversation.speak(content=f'{conversation.query}\n{action.raw_input}')
            conversation = conversation.listen(gpt=gpt)
            logger = logger.log(conversation)
        except openai.BadRequestError as e:
            print(f'[*] ')
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"[*] OpenAI API returned an API Error: {e}")
        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"[*] OpenAI API request exceeded rate limit: {e}")
        return gpt, conversation.breath(), logger

    # >exec
    def execute(self, gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Execute a system-wide command from within the program.
        Author's intended use case is directory traversal"""
        try:
            if action.arguments[0] == 'cd':  # hack to allow the user to change directories
                if action.arguments[1] == 'home':
                    os.chdir(logger.paths.cwd)
                else:
                    os.chdir(action.arguments[1])
                print(f'[*] cwd ~ {os.getcwd()}')
            elif action.arguments[0] == 'cat':
                print(self._fetch_contents(action.arguments[1]), '\n')
            else:
                output = subprocess.check_output(action.arguments[:], shell=True, text=True,
                                                 stderr=subprocess.STDOUT, timeout=3)
                print(output)
        except subprocess.CalledProcessError as e:
            print(f'[*] subprocess error: {e}')
        except OSError as e:
            print(f'[*] os error: {e}')
        return gpt, conversation, logger

    # >insert
    def insert(self, gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """While thinking: Appends the contents of a file to the query with the >insert command"""
        insert_me = self._fetch_contents(action.arguments[0])
        conversation.query += f'{conversation.query}\n{insert_me}'
        return gpt, conversation, logger

    # >flush
    @staticmethod
    def flush(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """While thinking: Resets the conversation.query to ''"""
        conversation = conversation.breath()
        return gpt, conversation, logger

    # >save
    @staticmethod
    def save(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Extract and save code, the reply, or the response object to an absolute, relative, or generic path"""
        try:
            logger.save(arguments=action.arguments, conversation=conversation)
        except FileNotFoundError:
            print(f'[*] error saving data')
        return gpt, conversation, logger

    # >set
    @staticmethod
    def set(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Allows the user to change the values for the keys of instantiated objects"""
        try:
            if action.arguments[0] == 'level' or action.arguments[0] == 'format':
                setattr(logger, action.arguments[0], ast.literal_eval(action.arguments[1]))
                print(f'[*] gpt ~ {action.arguments[0]}: {action.arguments[1]}')
            elif action.arguments[0] in ['model', 'temperature', 'top_p', 'n', 'frequency_penalty',
                                         'presence_penalty', 'max_tokens']:
                setattr(gpt, action.arguments[0], ast.literal_eval(action.arguments[1]))
                print(f'[*] gpt ~ {action.arguments[0]}: {action.arguments[1]}')
            elif action.arguments[0] == 'gpt':
                if action.arguments[1] == 'client':
                    print('[*] use `>reset client` to change API key')
                else:
                    setattr(gpt, action.arguments[1], ast.literal_eval(action.arguments[2]))
                    print(f'[*] {action.arguments[0]} ~ {action.arguments[1]}: {action.arguments[2]}')
            elif action.arguments[0] == 'logger' and ('format' == action.arguments[1] == 'level'):
                setattr(logger, action.arguments[1], ast.literal_eval(action.arguments[2]))
                print(f'[*] {action.arguments[0]} ~ {action.arguments[1]}: {action.arguments[2]}')
            else:
                print('[*] invalid entry')
        except AttributeError:
            print('[*] attribute error')
        except ValueError:
            print('[*] value error')
        except TypeError:
            print('[*] type error')
        return gpt, conversation, logger

    # >show
    @staticmethod
    def show(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Display values contained by objects: gpt, conversation, action, and logger"""
        try:
            if len(action.arguments) == 0:
                print(f'[*] query:\n{conversation.query}\n')
            elif action.arguments[0] == 'query':
                print(f'[*] query:\n{conversation.query}\n')
            elif action.arguments[0] == 'gpt' and len(action.arguments) == 2:
                print(f'[*] gpt ~ {action.arguments[1]}: {getattr(gpt, action.arguments[1])}\n')
            elif action.arguments[0] == 'conversation':
                print(f"[*] conversation ~ {action.arguments[1]}: {getattr(conversation, action.arguments[1])}\n")
            elif action.arguments[0] == 'logger':
                print(f'[*]logger ~ {action.arguments[1]}: {getattr(logger, action.arguments[1])}')
            elif action.arguments[0] == 'all':
                objects = [gpt, logger]
                for instance in objects:
                    print(f'\n[*] {type(instance).__name__}')
                    for key, value in instance.__dict__.items():
                        if key == 'client' or key == 'paths':
                            pass
                        else:
                            print(f"[-] {key.lstrip('_')}: {value}")
            elif action.arguments[0] == 'gpt':
                print("\n[*] GPT:")
                for key, value in gpt.__dict__.items():
                    if key == 'client':
                        pass
                    else:
                        print(f"[-] {key.lstrip('_')}: {value}")
            elif [action.arguments[0] in k for k, v in gpt.__dict__.items()]:
                print(f"[*] {action.arguments[0]}: {getattr(gpt, action.arguments[0])}")
        except AttributeError:
            print('[*] invalid entry')
        return gpt, conversation, logger

    # >reset
    @staticmethod
    def reset(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Reset the AI assistant, or start a new log entry"""
        if len(action.arguments) == 0 or ('chat' == action.arguments[0] == 'conversation'):
            print('[*] resetting AI')
            logger.new_log()
            prompt = SystemPrompt(prompts_dir=logger.paths.prompts)
            conversation = Conversation().start(prompt.content)
        elif action.arguments[0] == 'log':
            logger.new_log()
        return gpt, conversation, logger

    # >help
    @staticmethod
    def help(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """Prints the help string for the context management commands"""
        print(f'[*] hkrsAI\ncommand: >help\n{HELP_STRING}')
        return gpt, conversation, logger

    # >exit
    @staticmethod
    def goodbye(gpt: GPT, conversation: Conversation, action: Action, logger: Logger):
        """This is the end"""
        logger.log(conversation)
        print('\n[*] exiting\n')
        sys.exit()

    @staticmethod
    def _fetch_contents(file_path):
        """Return the contents of a file as a string variable"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            pass
        
