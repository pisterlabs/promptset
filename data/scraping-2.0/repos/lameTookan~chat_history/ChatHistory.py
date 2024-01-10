
import openai
import tiktoken 
import os 
import time
import random
import sys
import json
from datetime import datetime, timedelta
import MyStuff as ms 
import functions as fn 
import my_secrets as secrets
from collections import deque
from typing import Callable, List, Union, Tuple, Dict, Any, Optional
"""
to do 
[] Clean up code, improve documentation comments, and type hints 
[x] Make new unittest classes for the recently added functions
[] Impliment by none_or_value function thru out my code
[] Maybe move system prompt and reminder methods to another nested class
[x] Finish or scrap new search and edit methods
    []  Refactor the full chat_history if I want to keep these methods
[] Decide if the prompt shift feature is worth keeping or if it overcomplicates things
[x] Add an ability to edit the _make_message_pretty method to be configurable in the constructor(passing a function refrence as a parameter)
[x] Change the make message pretty class to not rely on my MyStuff module, add the colors directly in this method 
[] Test out saving and loading
[] Change the token counting system - store trimmed chat log as a list of tuples and dictionaries so we dont have to count each message every single iteration 
[] Make sure save file accomidates new deque strcuture 
[] 

"""
class RoleError(Exception):
    def __init__(self, role=None, allowed_roles=None):
        if role is not None or allowed_roles is not None:
            self.role = role
            self.allowed_roles = allowed_roles
            self.message = f"Role {self.role} not in allowed roles {self.allowed_roles}"
            super().__init__(self.message)
        else:
            super().__init__("Role not in allowed roles")
        
class NoSystemPromptError(Exception):
    def __init__(self, message=None):
       self.message = message if message is not None else "No system prompt set"    
    def __str__(self):
        return self.message

class MalformedSaveDictError(Exception):
    def __init__(self, missing_key= None):

        if missing_key is not None:
            message = "Save dict missing key: " + missing_key
        else:
            message = "Save dict is malformed"
        self.message = message
    def __str__(self):
        return self.message
class BadChatLogError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Chat log is malformed"
        super().__init__(message)
class BadFuncRefProvidedError(Exception):
    def __init__(self, message=None):
        prepend = "Function reference provided does not return expected data type: "
        if message is None:
            message = prepend
        else:
            message = prepend + message
        self.message = message
    def __str__(self):
        return self.message
        


class ChatHistory:
    default_allowed_roles = ['user', 'system', 'assistant']
    default_system_prompt_wildcards = {
        "date": {
            "description": "The current date",
            "value": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        },
        "cut_off_date": {
            "description": "The date training data was last updated",
            "value": "09/21/2021, 00:00:00"
        } ,  
        "shift": {
            "description": "A special dynamic wildcard that shifts the system prompt to the next n messages in the chat log",
            "value": 'shift'

        }
    }
    _val_if_not_None = lambda val, default: val if val is not None else default

    default_sys_prompt = "You are a helpful AI Assistant. It is currently {date}, and the training data was last updated on {cut_off_date}. {shift}"
    def __init__(self, 
            max_model_tokens= 8000, max_completion_tokens=1000, token_padding = 500, 
            chat_log_folder="chat_logs", allowed_roles = None,
            system_prompt_wildcards = None, token_counter_function = None, get_pretty_message_function = None):
        # replace _count_tokens with a function that takes a string and returns the number of tokens in that string during initialization if one is provided 
        self._count_tokens = token_counter_function if token_counter_function is not None else self._default_count_tokens
        # replace _get_pretty_message with a function that takes a message and returns a stylized version of that message during initialization if one is provided
        self._get_pretty_message = self._default_get_pretty_message if get_pretty_message_function is None else get_pretty_message_function

        self.allowed_roles = self.default_allowed_roles if allowed_roles is None else allowed_roles
      
        self.system_prompt_wildcards = self.default_system_prompt_wildcards if system_prompt_wildcards is None else system_prompt_wildcards
        self.max_model_tokens = max_model_tokens
        self.max_completion_tokens = max_completion_tokens
        self.full_chat_log = []
        self.trimmed_chat_log = deque()
        self._sys_prompt = None
        self.max_chat_log_tokens = None
        self.token_padding = token_padding
        self.trimmed_messages = 0
        self.is_loaded = False
        self.loaded_from = None 
        # will be set to true if the chat history is loaded from a save file
        self.prompt_shift = None
        self.shift_until = None
        self._reminder_value = None 
        
       
        self.save_dict = self.SaveDict(self)
        self.save_to_file = self.SaveToFile(self, chat_log_folder)
        self.constructor_args = {
            "max_model_tokens": max_model_tokens,
            "max_completion_tokens": max_completion_tokens,
            "token_padding": token_padding,
            "chat_log_folder": chat_log_folder,
            "allowed_roles": self.allowed_roles,
            "system_prompt_wildcards": self.system_prompt_wildcards,


        }

        

    class SaveDict:
        def __init__(self, chat_history: 'ChatHistory'):
            self.chat = chat_history
        def to_dict(self) -> dict:
            """Returns a dict containing all the information needed to save the chat history"""
            full_chat_log = self.chat.full_chat_log
            trimmed_chat_log = self.chat.trimmed_chat_log
            sys_prompt = self.chat._sys_prompt
            max_chat_log_tokens = self.chat.max_chat_log_tokens
            token_padding = self.chat.token_padding
            max_model_tokens = self.chat.max_model_tokens
            max_completion_tokens = self.chat.max_completion_tokens
            save_dict = {
                "general_info": {
                    "date": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                    "chat_log_folder": self.chat.save_to_file.chat_log_folder,
                    "allowed_roles": self.chat.allowed_roles,
                    "system_prompt_wildcards": self.chat.system_prompt_wildcards,
                    "token_counter_function": self.chat._count_tokens.__name__
                    
                },
                "sys_prompt": sys_prompt,
                "full_chat_log": full_chat_log,
                "trimmed_chat_log": trimmed_chat_log,
                "token_info": {
                    "max_chat_log_tokens": max_chat_log_tokens,
                    "token_padding": token_padding,
                    "max_model_tokens": max_model_tokens,
                    "max_completion_tokens": max_completion_tokens

                }
            }
            if self.chat.is_loaded:
                save_dict["loaded_from"] = self.chat.loaded_from
            if self.chat.prompt_shift is not None:
                save_dict["prompt_shift"] = {
                    "shift_until": self.chat.shift_until,
                    "value": self.chat.prompt_shift
                }
            else:
                save_dict["prompt_shift"] = None
            if self.chat._reminder_value is not None:
                save_dict["reminder_value"] = self.chat._reminder_value
            else:
                save_dict["reminder_value"] = None
            
            return save_dict
        def load_from_dict(self, save_dict: dict) -> None:
            self.check_save_dict(save_dict)
            self.chat._sys_prompt = save_dict["sys_prompt"]
            self.chat.full_chat_log = save_dict["full_chat_log"]
            self.chat.trimmed_chat_log = save_dict["trimmed_chat_log"]
            self.chat.max_chat_log_tokens = save_dict["token_info"]["max_chat_log_tokens"]
            self.chat.token_padding = save_dict["token_info"]["token_padding"]
            self.chat.max_model_tokens = save_dict["token_info"]["max_model_tokens"]
            self.chat.max_completion_tokens = save_dict["token_info"]["max_completion_tokens"]
            self.chat.is_loaded = True
            self.chat.loaded_from = save_dict["loaded_from"] if "loaded_from" in save_dict else None
            self.chat.prompt_shift = save_dict["prompt_shift"]["value"] if save_dict["prompt_shift"] is not None else None
            self.chat.shift_until = save_dict["prompt_shift"]["shift_until"] if save_dict["prompt_shift"] is not None else None
            self.chat._reminder_value = save_dict["reminder_value"] if "reminder_value" in save_dict else None
       

        def check_save_dict(self, save_dict: dict) -> None:
            "Checks if save_dict is valid, raises MalformedSaveDictError if not"
            if not isinstance(save_dict, dict):
                raise TypeError("save_dict must be a dict")
            if not "sys_prompt" in save_dict:
                raise MalformedSaveDictError("sys_prompt")
            if not "token_info" in save_dict:
                raise MalformedSaveDictError("token_info")
            else: 
                for key in ["max_model_tokens", "max_completion_tokens", "token_padding"]:
                    if not key in save_dict["token_info"]:
                        raise MalformedSaveDictError(key)
            if not "full_chat_log" in save_dict:
                raise MalformedSaveDictError("full_chat_log")
            if not "trimmed_chat_log" in save_dict:
                raise MalformedSaveDictError("trimmed_chat_log")
            if not "trimmed_messages" in save_dict:
                raise MalformedSaveDictError("trimmed_messages")
    class SaveToFile:
        def __init__(self, chat_history: 'ChatHistory', chat_log_folder: str):
            self.chat = chat_history
            self.save_dict = self.chat.save_dict
            self.chat_log_folder = chat_log_folder
        def save(self, file_path: str, overwrite = False) -> None:
            """Saves chat history to file"""
            file_path = self._add_file_path(file_path)
            save_dict = self.save_dict.to_dict()
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            if os.path.exists(file_path) and not overwrite:
               return False
            with open(file_path, "w") as f:
                json.dump(save_dict, f)
            return True
        

        def load(self, file_path: str) -> bool:
            """Loads history from file, if file does not exist returns False, returns True if successful"""
            file_path = self._add_file_path(file_path)
            if not os.path.exists(file_path):
               return False
            with open(file_path, "r") as f:
                save_dict = json.load(f)
            self.save_dict.load_from_dict(save_dict)
            self.chat.loaded_from = file_path
            return True
        def _add_file_path(self, file_name: str) -> str:
            """Takes a file name and returns a file path, if file name does not end with .json, adds it, if file name does not start with chat_log_folder, adds it"""
            if not file_name.endswith(".json"):
                file_name += ".json"
            if not file_name.startswith("/"+ self.chat_log_folder + "/"):
               os.path.join(self.chat_log_folder, file_name)
            return file_name
        def get_pretty_file_name(self, file_name: str) -> str:
            """Takes a filename or path and returns a pretty file name
            What is a pretty file name?
            A pretty file name is a file name that does not contain the chat log folder name or the .json extension

            """
            if file_name.startswith("/"+ self.chat_log_folder + "/"):
                file_name = os.path.basename(file_name)
            if file_name.endswith(".json"):
                file_name = file_name[:-5]
            return file_name
        def get_save_file_names(self, pretty = False, as_list = True) -> list:
            """Returns a list of save files in the chat log folder"""
            filenames = os.listdir(self.chat_log_folder)
            if pretty:
                filenames = [self.get_pretty_file_name(filename) for filename in filenames]
            if as_list:
                return filenames
            else:
                return "\n".join(filenames)
        def check_if_file_exists(self, file_name: str) -> bool:
            """Checks if a file exists"""
            file_name = self._add_file_path(file_name)
            return os.path.exists(file_name)
    
    def clear(self, clear_token_info: bool = False ) -> None:
        """Clears chat history"""
        self.chat_log = []
        self.trimmed_chat_log = []
        self.trimmed_messages = 0
        self.is_loaded = False
        self.loaded_from = None
        self.reminder = None
        self.sys_prompt = None
        self.prompt_shift = None
        if clear_token_info:
            self.modify_token_info(0, 0, 0)
        
    def modify_token_info(self, max_model_tokens: int = None, max_completion_tokens: int = None, token_padding: int = None) -> None:
        """Modifies token info and works out tokens"""
        self.max_chat_log_tokens = self._val_if_not_None(max_model_tokens, self.max_chat_log_tokens)
        self.max_model_tokens = self._val_if_not_None(max_model_tokens, self.max_model_tokens)
        self.max_completion_tokens = self._val_if_not_None(max_completion_tokens, self.max_completion_tokens)
        self.token_padding = self._val_if_not_None(token_padding, self.token_padding)
        self.work_out_tokens()
        self.trim_chat_log()

    def _default_count_tokens(self, text: str, model = "gpt-4") -> int:
            """Returns the number of tokens in a string"""
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
    
    def _count_tokens_in_chat_log(self, chat_log: list) -> int:
        """Returns the number of tokens in a chat log"""
        total = 0 
        for message in chat_log:
            total += self._count_tokens(message['content'])
        return total
  
    def _format_message (self, message: str, role: str) -> dict:
        """Format a message to be added to the chat log"""
        if role not in self.allowed_roles:
            raise RoleError(role, self.allowed_roles)
        return {
            "role": role,
            "content": message,
        }
    def _check_token_counter(self, func_ref) -> None:
        val = func_ref('test')
        if not isinstance(val, int):
            raise TypeError("Token counter function must return an int")
    def _default_get_pretty_message(self, message:dict) -> str:
        """Gets a pretty verison of a message
        What is pretty?
        User messages are prefixed with a >, and uncoloured
        System messages are prefixed with a >>>, and coloured yellow
        Assistant messages are prefixed with a >>, and coloured cyan
        Please, note that this only the *default* get_pretty_message, you can change this by passing a get_pretty_message function to the ChatHistory constructor
        """
        role = message['role']
        
        pretty_message = ""
        
        if message['role'] == 'user':
            pretty_message = "> " + message['content']
        elif message['role'] == 'system':
            pretty_message = ">>> " + "\u001b[33m"+message['content'] +" \u001b[0m"
        elif message['role'] == 'assistant':
            pretty_message = ">> " + " \u001b[36m"+ message['content']+ "\u001b[0m"
        return pretty_message
    def _check_message_dict(self, message: dict) -> None:
        if not isinstance(message, dict):
            raise TypeError("Message must be a dict")
        if not "role" in message.keys():
            raise BadChatLogError("A role is required")
        if message['role'] not in self.allowed_roles:
            raise RoleError(message['role'], self.allowed_roles)
        if not "content" in message.keys():
            raise BadChatLogError("A content is required")
    def add_message_list(self, chat_list: list) -> None:
        """Loads a chat log from a list"""
        for message in chat_list:
            self._check_message_dict(message)
            self.add_message(message['content'], message['role'])

    def save(self, file_name, overwrite=False) -> None:
        """Calls the ChatHistory.save_to_file.save method"""
        return self.save_to_file.save(file_name, overwrite)
    def load(self, file_name) -> None:
        """Calls the ChatHistory.save_to_file.load method"""
        return self.save_to_file.load(file_name)
    

            
    def check_sys_prompt(self) -> None:
        """Check that the system prompt is not None. If it is, raise an error"""
        if self._sys_prompt is None:
            raise NoSystemPromptError
    
    

    def add_sys_wildcards(self, text, ) -> str:
        """Adds wildcards to the given string"""
        wildcards = {name: value['value']  for name, value in self.system_prompt_wildcards.items()}
        wildcards['shift'] = self.prompt_shift if self.prompt_shift is not None else " "
        return text.format(**wildcards)
    @property
    def sys_prompt(self) -> str:
        """Return the system prompt, with the wildcards filled in"""
        return self.add_sys_wildcards(self._sys_prompt)
    @sys_prompt.setter
    def sys_prompt(self, value: str)-> None:
        """Set the system prompt, and work out how many tokens are left for the chat log"""
        if not isinstance(value, str) or value == "default":
            value = self.default_sys_prompt
        self._sys_prompt = value
        self.work_out_tokens()
    @property
    def reminder (self) -> str:
        """Returns the value of a reminder"""
        if  not self._reminder_value == None :
            return "Reminder" + self.add_sys_wildcards(self._reminder_value)
        else:
            return None
    @reminder.setter
    def reminder(self, value: str) -> None:
        """Sets the value of a reminder"""
       
            
        self._reminder_value = value

        self.work_out_tokens()
        self.trim_chat_log()
    
    
    def work_out_tokens (self) -> None:
        """Work out how many tokens are left for the chat log after the system prompt and the padding are taken into account"""
        self.check_sys_prompt()
        sys_prompt_tokens = self._count_tokens(self.sys_prompt)
        reminder_tokens = 0 if self.reminder is None else self._count_tokens(self.reminder)
        shift_tokens = 0 if self.prompt_shift is None else self._count_tokens(self.prompt_shift)
        self.max_chat_log_tokens = self.max_model_tokens -(sys_prompt_tokens + self.token_padding + self.max_completion_tokens + reminder_tokens + shift_tokens)
    def add_message(self, message: str, role: str) -> None:
        """Add a message to the chat log"""
        if role == "assistant":
            self.shift_tick()
        self.check_sys_prompt()
        formatted_message = self._format_message(message, role)
        self.full_chat_log.append(formatted_message)
        self.trimmed_chat_log.append(formatted_message)
        self.trim_chat_log()
    def trim_chat_log(self) -> None:
        """Trim the chat log to the maximum number of tokens allowed"""
        self.check_sys_prompt()
        """Trim the chat log to the maximum number of tokens allowed"""
        while self._count_tokens_in_chat_log(self.trimmed_chat_log) > self.max_chat_log_tokens:
            self.trimmed_chat_log.popleft()
            self.trimmed_messages += 1
    @property
    def finished_chat_log(self) -> list:

        """Return the chat log with the system prompt at the start"""
        self.check_sys_prompt()

        formatted_system_prompt = self._format_message(self.sys_prompt, "system")
        finished = []
        if self.reminder != None:
            formatted_reminder = self._format_message(self.reminder, "system")
            finished = [formatted_system_prompt] + list(self.trimmed_chat_log) + [formatted_reminder]
        else:
            finished = [formatted_system_prompt]+ list(self.trimmed_chat_log)
        return finished 
    def get_messages (self, n = 1, role = None, pretty = False, reverse = False) -> Union[list, dict, str]:
        self.check_sys_prompt()
        """Get the last n messages in the chat log. If role is specified, only get messages from that role. If pretty is True, return a string with the messages formatted nicely"""
        messages = []
        if role is None:
            messages = self.full_chat_log[:-n]
        else:
           found_messages_numb = 0 
           for message in self.full_chat_log:
                if message['role'] == role:
                   messages.append(message)
                   found_messages_numb += 1
                if found_messages_numb == n:
                    break
        if len(messages) == 0:
            return None
        elif n == 1:
            messages = messages[0]
            print(messages)
            if pretty:
                messages = self._get_pretty_message(messages)
            return messages
        elif pretty:
            messages = [self._get_pretty_message(message) for message in messages]
            if reverse:
                messages.reverse()
            messages = "\n".join(messages)
            return messages
        else:
            if reverse:
                messages.reverse()

            return messages
    @property 
    def assistant_message(self) -> str:
        """Get the last assistant message in the chat log"""
        self.check_sys_prompt()
        message = self.get_messages(role="assistant", pretty=True)
        print(message)
        return message
    @assistant_message.setter 
    def assistant_message(self, value) -> None:

        """Add a message to the chat log with the role assistant"""
        self.check_sys_prompt()
        self.add_message(value, "assistant")
    @property
    def user_message(self) -> str:
        """Get the last user message in the chat log"""
        return self.get_messages(role="user", pretty=True)
    @user_message.setter
    def user_message(self, value)-> None:
        """Add a message to the chat log with the role user"""
        self.check_sys_prompt()
        self.add_message(value, "user")
    # prompt shift system 
    def set_prompt_shift(self, value, until = None) -> None:
        """Sets {shift} wildcard to the given value, and sets the prompt shift to expire after until ticks, or never if until is None"""
        self.prompt_shift = value
        self.shift_until = until
        self.work_out_tokens()
        self.trim_chat_log()
    def shift_tick (self) -> None:
        """Tick down the prompt shift, and remove it if it has expired"""
        if self.shift_until is not None:
            self.shift_until -= 1
            if self.shift_until == 0:
                self.prompt_shift = None
                self.shift_until = None
                self.work_out_tokens()
                self.trim_chat_log()

    def __str__(self)-> str:
        """Return a pretty version of the chat log"""
        return self.get_messages(pretty=True)
    def __repr__(self):
        """Return a string that can be used to recreate the chat log"""
        """
        refrence for arg dictionary:
        self.constructor_args = {
            "max_model_tokens": max_model_tokens,
            "max_completion_tokens": max_completion_tokens,
            "token_padding": token_padding,
            "chat_log_folder": chat_log_folder,
            "allowed_roles": self.allowed_roles,
            "system_prompt_wildcards": self.system_prompt_wildcards,


        }

        """
        

        constructor = "ChatHistory({max_model_tokens}, {max_completion_tokens}, {token_padding}, {chat_log_folder}, {allowed_roles}, {system_prompt_wildcards})".format(**self.constructor_args)
        important_vars = """
        Important Variables:
        reminder = {reminder}
        sys_prompt = {sys_prompt}
        is_loaded = {is_loaded}
        loaded_from = {loaded_from}
        system_prompt_wildcards = {system_prompt_wildcards}
        allowed_roles = {allowed_roles}
        shift = {shift} until {shift_until}
        """.format(sys_prompt = self.sys_prompt, is_loaded = self.is_loaded, loaded_from = self.loaded_from, shift = self.prompt_shift, shift_until = self.shift_until if not self.shift_until == None else "Forever" , **self.constructor_args)
        return constructor + important_vars
    def __len__(self):
        """Return the number of messages in the chat log"""
        return len(self.trimmed_chat_log)
    
class TimeContextManager:
    def __init__(self):
        self.time_start = None
        self.time_end = None
        self.time_taken = None
    def __enter__(self):
        self.time_start = time.time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.time_end = time.time()
        self.time_taken = self.time_end - self.time_start
        print("Time taken: {0:.2f} seconds".format(self.time_taken))
        return False
    
