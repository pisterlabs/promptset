import datetime
import json
import os
import random
import sys
from collections import namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import openai
import tiktoken

import from_file.from_file as ff
import misc.MyStuff as ms
import object_factory as fact
from ExportChatLogs import export_chat_menu
from settings import (API_KEY, BYPASS_MAIN_MENU, DEFAULT_MODEL,
                      DEFAULT_TEMPLATE_NAME)

def get_default_template_for_model(model: str) -> str:
    if model == "gpt-3":
        return "gpt-3_default"
    elif model == "gpt-4":
        return "gpt-4_default"
    else:
        return "gpt-3_default"
    


def split_input_from_cmd(cmd: str, string: str) -> str:
    if cmd in string:
        return string.split(cmd)[1].strip()
    else:
        return None


def format_dict_as_string(d: dict, delim: str = ": ", title_case=False) -> str:
    """Formats a dictionary as a string with each key-value pair on a new line, separated by a delimiter.
    Does the same for nested dictionaries. Nested tuples, lists, and sets are also supported are separated by commas.
    """

    result = []
    for key, value in d.items():
        key = key.title() if title_case else key
        if isinstance(value, dict):
            item = []
            item.append(key)
            for k, v in value.items():
                k = k.title() if title_case else k
                item.append(f">>{k}{delim}{v}")

            value = "\n".join(item)
        elif (
            isinstance(value, list)
            or isinstance(value, tuple)
            or isinstance(value, set)
        ):
            value = [key] + [f"{v}" for v in value]
            value = ", ".join(value)
        else:
            value = f"{key}{delim}{value}"
        result.append(value)
    return "\n".join(result)


def toggle(input: bool):
    if input == True:
        return False
    else:
        return True


def bool_to_yes(val: bool, y="yes", n="no") -> str:
    if val:
        return y
    else:
        return n


def chunk_input(ini_message: str = "Type enter twice when done.", input_message="> "):
    print(ini_message)
    input_list = []
    while True:
        ans = input(input_message)
        if ans == "":
            break
        input_list.append(ans)
    return " ".join(input_list)


def confirm(message: str = "Are you sure?", y_n="(y/n)", y="y", n="n") -> bool:
    print(message + " " + y_n)
    while True:
        ans = input("> ")
        if ans.lower() == y:
            return True
        elif ans.lower == n:
            return False
        else:
            print("Invalid input. Please try again.")


class SystemPromptManager:
    """
    Class for managing system prompts, including saving, loading, and modifying them from a text file
    Attributes:
        sys_info_dict (dict): A dictionary of wildcards to replace in the system prompts, from the ChatLog class
        save_folder (str): The folder to save the system prompts in
    Methods:
        add_file_path(file_name: str) -> str: Adds a file path and file extension to the file name if it doesn't already have one
        remove_file_path(file_name: str) -> str: Removes the file path and file extension from the file name if it has one
        read_file(file_name: str) -> str: Reads a file and returns the contents
        write_file(file_name: str, text: str, overwrite: bool = False) -> bool: Writes the contents to a file. If overwrite is True, overwrites the file if it exists. Otherwise, appends to the file Output is True if the file was written to successfully, False otherwise
        get_file_names(remove_filenames = True) -> list[str]: Returns a list of all the file names in the save folder, removes the file path and file extension if  remove_filenames is True
        get_wild_card_info(self): Returns a string containing the wildcards and their descriptions
    Example Usage:
        spm = SystemPromptManager()
        spm.write_file("test.txt", "Hello, world!")
        spm.read_file("test.txt")

    """

    def __init__(
        self,
        sys_info_dict=fact.cw.g.ch.ChatLog.system_prompt_wildcards,
        save_folder="system_prompts",
    ):
        self.sys_info_dict = sys_info_dict
        save_folder = (
            save_folder + "/" if not save_folder.endswith("/") else save_folder
        )
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.save_folder = save_folder

    def add_file_path(self, file_name: str) -> str:
        """
        Adds a file path and file extension to the file name if it doesn't already have one
        """
        if not file_name.endswith(".txt"):
            file_name = file_name + ".txt"
        if not file_name.startswith(self.save_folder):
            file_name = self.save_folder + file_name
        return file_name

    def remove_file_path(self, file_name: str) -> str:
        """
        Removes the file path and file extension from the file name if it has one, getting just the file name
        """
        if file_name.startswith(self.save_folder):
            file_name = file_name[len(self.save_folder) :]
        if file_name.endswith(".txt"):
            file_name = file_name[:-4]
        return file_name

    def read_file(self, file_name: str) -> str:
        """Reads a file and returns the text as a string"""
        file_name = self.add_file_path(file_name)
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist")
        try:
            with open(file_name, "r") as f:
                file = f.read()
            return file
        except:
            raise Exception(f"Could not read file {file_name}")

    def write_file(self, file_name: str, text: str, overwrite=False) -> bool:
        """
        Writes a file with the given text, overwriting if overwrite is True. Outputs True if the file was written, False if not
        _summary_

        Args:
            file_name (str): the name of the file to write to
            text (str): the text to write to the file
            overwrite (bool, optional): If False, file will be overwritten if it already exists . Defaults to False.

        Returns:
            bool: False if the file was not written, True if it was
        """
        file_name = self.add_file_path(file_name)
        if os.path.exists(file_name) and not overwrite:
            return False
        with open(file_name, "w") as f:
            f.write(text)
        return True

    def get_file_names(self, remove_file_path: bool = True) -> List[str]:
        """Gets a list of all file names in the save folder, optionally removing the file path and extension from the file name
        _summary_

        Returns:
            _type_: _description_

        """
        if remove_file_path:
            return [
                self.remove_file_path(file_name)
                for file_name in os.listdir(self.save_folder)
            ]
        else:
            return os.listdir(self.save_folder)

    def get_wildcard_info(self) -> str:
        """
        Using the system_prompt_wildcards dictionary, returns a string with all the wildcards and their descriptions
        """
        result = []
        for key, value in self.sys_info_dict.items():
            result.append(key + ": " + value["description"])
        return "\n".join(result)


class SysPromptManagerMenu:
    """Wrapper class for the SystemPromptManager class, with a menu for loading, viewing, and removing system prompts
    Methods:
        load_menu() -> None: Loading menu for system prompts

    """

    def __init__(self, sys_prompt_manager: SystemPromptManager):
        self.sys_prompt_manager = sys_prompt_manager
        self.loaded_file = None

    def load_menu(self):
        msg_list = [
            "Welcome to the System Prompt Loading Menu!",
            "Type 'help' to see this message again.",
            "Type 'list' to see a list of all available system prompts.",
            "Type view {file_name}' to view a system prompt.",
            "Type 'load {file_name}' to load a system prompt.",
            "Type remove {file_name}' to remove a system prompt.",
            "Type quit to return to the main menu.",
        ]
        message = "\n".join(msg_list)
        files_list = self.sys_prompt_manager.get_file_names()

        print(message)
        file_msg = (
            "System Prompts in Folder:"
            + "------\n"
            + "\n".join(files_list)
            + "\n"
            + "------"
        )
        while True:
            ans = input("> ")
            ans_lower = ans.lower()
            ans_lower = ans_lower.strip()
            if ans_lower in ("q", "quit", "exit"):
                if confirm("Are you sure you want to quit?"):
                    print("Returning to main menu...")
                    return None
                else:
                    continue
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("list", "ls", "l"):
                print(file_msg)
            elif ans_lower.startswith("view"):
                filename = split_input_from_cmd("view", ans)
                if filename is None or filename not in files_list:
                    print("Invalid file name. Please try again.")
                    continue
                else:
                    print(self.sys_prompt_manager.read_file(filename))
            elif ans_lower.startswith("load"):
                print("Loading system prompt...")
                cmd = split_input_from_cmd("load", ans)
                try:
                    loaded_file = self.sys_prompt_manager.read_file(cmd)
                    print("System prompt loaded successfully!")
                    self.loaded_file = loaded_file
                    return loaded_file
                except FileNotFoundError:
                    print("Invalid file name. Please try again.")
                    continue

            elif ans_lower.startswith("remove"):
                if split_input_from_cmd("remove", ans) in files_list:
                    if confirm("Are you sure you want to remove this system prompt?"):
                        os.remove(self.sys_prompt_manager.add_file_path(filename))
                        print("System prompt removed successfully!")
                    else:
                        print("System prompt not removed.")
                else:
                    print("Invalid file name. Please try again.")
                    continue
            else:
                print("Invalid command. Please try again.")

    def save_menu(self):
        """Menu for saving system prompts"""
        msg_list = [
            "Welcome to the System Prompt Saving Menu!",
            "Type 'help' to see this message again.",
            "Type 'list' to see a list of all available system prompts.",
            "Type 'view {file_name}' to view a system prompt.",
            "Type quit to return to the main menu.",
            "Type remove {file_name}' to remove a system prompt.",
            "Please note that you can also save new system prompts by adding them to the system_prompts folder, as .txt files."
            "Otherwise, type  a file name to save a new system prompt. ",
        ]

        def refresh_file_list() -> tuple[str, list]:
            """Returns a string with all the file names in the save folder"""
            return (
                "\n".join(self.sys_prompt_manager.get_file_names()),
                self.sys_prompt_manager.get_file_names(),
            )

        files, files_list = refresh_file_list()
        message = "\n".join(msg_list)
        print(message)
        print("System Prompts in Folder:" + f"------\n {files}  \n ------")
        while True:
            files, files_list = refresh_file_list()
            ans = input("> ")
            ans_lower = ans.lower()
            if ans_lower in ("q", "quit"):
                if confirm("Are you sure you want to quit?"):
                    print("Returning to main menu...")
                    break
                else:
                    continue
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("list", "ls", "l"):
                print(files)
            elif ans_lower.startswith("view"):
                filename = split_input_from_cmd("view", ans)
                if filename is None or filename not in files_list:
                    print("Invalid file name. Please try again.")
                    continue
                else:
                    print(self.sys_prompt_manager.read_file(filename))
            elif ans_lower.startswith("remove"):
                filename = split_input_from_cmd("remove", ans)
                if filename is None or filename not in files_list:
                    print("Invalid file name. Please try again.")
                    continue
                else:
                    if confirm(f"Are you sure you want to delete {filename}?"):
                        os.remove(self.sys_prompt_manager.add_file_path(filename))
                        print("System prompt removed successfully!")
                    else:
                        print("System prompt not removed.")
                        continue
            else:
                if confirm(
                    f"Are you sure you want to write a new system prompt to filename {ans}?"
                ):
                    if ans in files_list:
                        if confirm(f"Are you sure you want to overwrite {ans}?"):
                            over_write = True

                        else:
                            print("System prompt not saved.")
                            continue
                    else:
                        over_write = False
                    txt = self.write_file_menu(ans)
                    if txt is False:
                        continue
                    else:
                        self.sys_prompt_manager.write_file(ans, txt, over_write)
                        print("System prompt saved successfully!")

    def write_file_menu(
        self,
        filename: str,
    ) -> str:
        """Menu for writing a system prompt to a file"""
        print("To quit without saving, enter 'q' or 'quit'.")
        while True:
            print("You can use the following wildcards in your system prompt:")
            print(self.sys_prompt_manager.get_wildcard_info())
            ans = chunk_input(
                "Please input the text you would like to save. Double tap enter to save."
            )
            if ans is None:
                print("Invalid input. Please try again.")
            if ans.lower() in ("q", "quit"):
                if confirm("Are you sure you want to quit?"):
                    print("Returning to main menu...")
                    return False
                else:
                    continue
            else:
                print(
                    "Are you sure you want to save the following system prompt as "
                    + filename
                    + "?"
                )
                print(ms.yellow(ans))
                confirm = input("y/n > ").lower()
                if confirm.lower() in ("y", "yes"):
                    return ans

                else:
                    print(
                        "System prompt not saved. Type 'q' or 'quit' to quit without saving."
                    )
                    continue

    def main_menu(self, from_chatloop=True) -> None | str:
        """Main menu for the system prompt manager"""
        if self.loaded_file is not None:
            print("Loaded system prompt: " + str(self.loaded_file))
        msg_list = [
            "Welcome to the System Prompt Manager!",
            "Type 'help' to see this message again.",
            "Type 'save' to save a system prompt.",
            "Type 'quit' to exit the manager.",
        ]
        if from_chatloop:
            msg_list.append("Type 'load' to load a system prompt.")

        message = "\n".join(msg_list)
        print(message)
        while True:
            ans_lower = input("> ").lower()
            if ans_lower in ("q", "quit"):
                if confirm("Are you sure you want to quit?"):
                    print("Exiting...")
                    return None
                else:
                    continue
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("load", "l") and from_chatloop:
                prompt = self.load_menu()
                if prompt is not None:
                    print("Loaded system prompt: " + prompt)
                    print("Exiting system prompt manager...")
                    self.loaded_file = prompt
                    return prompt
                else:
                    print("No system prompt loaded. Type quit to exit.")
                    continue
            elif ans_lower in ("save", "s"):
                print("Entering save menu...")
                self.save_menu()
                print("Returning to main system prompt manager menu...")
                continue
            else:
                print("Invalid command. Please try again.")

    def quick_sys_prompt(self, save=True):
        """Quickly change the system prompt, for use in the chatloop"""
        print("Enter a new system prompt. Double tap enter to set the prompt.")
        print("To quit without saving, enter 'q' or 'quit'.")
        print(
            "Type 'wildcards' to see a list of wildcards you can use in your system prompt."
        )
        print("Type 'save' to toggle saving the system prompt.")
        if save:
            print(
                "You will then be prompted to save the system prompt as a file. To load the new system prompt without saving, type 'q' or 'quit'."
            )
        while True:
            ans = chunk_input("New system prompt > ")
            ans_lower = ans.lower()
            ans_lower = ans_lower.strip()
            if ans_lower in ("quit", "q", "exit"):
                print("Returning to previous menu...")
                return None
            elif ans_lower in ("wild", "wildcards"):
                print(self.sys_prompt_manager.get_wildcard_info())
                continue
            elif ans_lower in ("save", "s"):
                save = toggle(save)
                if save:
                    print(
                        "System prompt will be saved, type 'save' again to disable saving."
                    )
                else:
                    print(
                        "System prompt will not be saved, type 'save' again to enable saving."
                    )
                continue
            else:
                print("Are you sure you want to set the system prompt to:")
                print(ms.yellow(ans))
                if confirm("Set system prompt to this? Or try again?"):
                    if save:
                        print("Currently saved system prompts:")
                        print("\n".join(self.sys_prompt_manager.get_file_names()))
                        while True:
                            file_name = input(
                                "Please enter a file name to save the system prompt as > "
                            ).strip()
                            if self.sys_prompt_manager.write_file(file_name, text=ans):
                                print("System prompt saved successfully!")
                                print("Returning to chat loop...")
                                return ans
                            else:
                                print("File name already exists.")
                                if confirm(
                                    "Are you sure you want to overwrite it? This cannot be undone"
                                ):
                                    self.sys_prompt_manager.write_file(
                                        file_name, text=ans, overwrite=True
                                    )
                                    print("System prompt saved successfully!")
                                    print("Returning to chat loop...")
                                    return ans
                                else:
                                    print("Please enter a new file name.")
                                    continue
                    else:
                        return ans
                else:
                    continue

    def get_wildcards(self) -> str:
        """Returns a string containing the wildcards"""
        return self.sys_prompt_manager.get_wildcard_info()


class ChatLoop:

    """
    This class is a freaking monster, and a mess.
    I don't expect to ever really be making other command line interfaces, nor do I expect to ever reuse this code, so I'm leaving it as is. It works, and works well, and does a good job of showing off the capabilities of the chat wrapper and this codebase in general. I recommend using a better library for creating a command line interface, or just a GUI.

    This class is the main chat loop for the chat wrapper. It allows users to chat with the model, as well as customizing every aspect of the implementation
    Attributes:
        sys_manager_obj: The system prompt manager object, used to change the system prompt
        chat_wrapper: The chat wrapper object, used to chat with the model
        gpt_chat: The gpt chat object,
        chat_log: The chat log object
        chunking: Whether or not to chunk the input into multiple inputs, to get around the 1024 char limit in many terminals
    Methods:
        run_chat_loop: The main chat loop for the chat wrapper
        load_menu: The menu for loading a saved chat log
        save_menu: The menu for saving a chat log
        modify_params: The menu for modifying the model parameters on the fly

    Example Usage:
        chat_wrapper = fact.wrapper_factory.make_chat_wrapper()
        sys_manager_obj = SysPromptManagerMenu()
        chat_loop = ChatLoop(sys_manager_obj, chat_wrapper)
        chat_loop.run_chat_loop()
    """

    def __init__(
        self,
        sys_manager_obj: SysPromptManagerMenu,
        chat_wrapper: fact.cw.ChatWrapper,
        chunking=True,
        file_selector: ff.FromFile = ff.file_selector,
    ):
        self.file_selector = file_selector
        self.chat_wrapper: fact.cw.ChatWrapper = chat_wrapper
        self.chunking = chunking
        self.model_name = chat_wrapper.chat_log.model if chat_wrapper.chat_log else DEFAULT_MODEL
        self.chat_log: fact.cw.g.ch.ChatLog = chat_wrapper.chat_log
        self.gpt_chat: fact.cw.g.GPTChat = chat_wrapper.gpt_chat
        self.sys_manager_obj: SysPromptManagerMenu = sys_manager_obj

    def from_file_cmd(self, cmd: str = None):
        help_list = [
            "The from_file command allows you to load a text file and use it as input to the model.",
            "You can either read the default.txt filename in the from_file folder or you can read a file of your choice from the from_file/files folder.",
            "Supported Commands",
            f"{ms.yellow('from_file')} - Reads the default.txt file in the from_file folder",
            f"{ms.yellow('from_file <filename>')} - Reads the specified file in the from_file/files folder. If not found, an error message will be displayed(without the program crashing or anything like that)",
            f"{ms.yellow('from_file menu')} - Will open up the from file menu, allowing you to select a file from the from_file/files folder",
            f"{ms.yellow('from_file help')} - Displays this message",
        ]
        help_message = "\n".join(help_list)
        if cmd is not None:
            command = cmd

            if command.lower().strip() == "help":
                print(help_message)
                return None
            elif command.lower().strip() == "menu":
                print("Opening from file menu...")
                file = self.file_selector.menu()
                print("File loaded successfully!")
                return file
            else:
                try:
                    file = self.file_selector.get_file(command)
                    print("File loaded successfully!")
                    return file
                except FileNotFoundError:
                    print("File not found, please try again.")

        else:
            print("Loading default.txt file...")
            return self.file_selector.get_default()

    def run_chat_loop(self):
        """Main chat loop for the chat wrapper"""
        quick_msg_list = [
            "Type quit to quit (and return to main menu)",
            "Type help to see all commands" "Type save to save the chat log",
            "Type load to load a chat log from the save folder",
        ]
        msg_list = [
            f"Type {ms.yellow('quit')} to quit (and return to main menu)",
            f"Type {ms.yellow('save')} to save the chat log",
            f"Type {ms.yellow('load')} to load a chat log from the save folder",
            f"Type {ms.yellow('params')} to modify the model parameters",
            f"Type {ms.yellow('clear')} to clear the chat log",
            f"Type {ms.yellow('help')} to see this message again",
            f"Type {ms.yellow('chunk')} to turn off/on input chunking (useful in linux/macOS)",
            f"Type {ms.yellow('from_file help')} to learn more about the read file from text file feature",
            f"Type {ms.yellow('sys')} to load or change a system prompt with saving enabled",
            f"Type {ms.yellow('quicksys')} to load or change a system prompt without saving",
            f"Type {ms.yellow('sysmanage')} to access the System Prompt Manager Menu",
            f"Type {ms.yellow('export')} to export the chat log to a text file(experimental). Save the chat log first!",
            f"Type {ms.yellow('print')} to print the full chat log to the console. "
        ]

        message = "\n".join(msg_list)
        print(
            "Welcome to the Chat Loop!",
        )
        print("\n".join(quick_msg_list))

        while True:
            if self.chunking:
                ans = chunk_input()
            else:
                ans = input("> ")

            ans_lower = ans.lower()
            ans_lower = ans_lower.strip()

            if ans_lower in ("quit", "exit", "q"):
                if confirm("Are you sure you want to quit?"):
                    print("Quitting...")
                    break
            elif ms.xy(ans_lower):
                pass
            elif ans_lower in ("save", "s"):
                self.save_menu()
                print("Returning to the chat loop...")
                continue
            elif ans_lower in ("export", "e"):
                print("Opening export menu...")
                export_chat_menu.main_menu()
                print("Back in the chat loop...")
            elif ans_lower.startswith("from_file"):
                command = ans_lower.replace("from_file", "").strip()
                if command == "" or command is None:
                    file_text = self.from_file_cmd()
                else:
                    file_text = self.from_file_cmd(command)
                if file_text is not None:
                    tokens = self.chat_wrapper.chat_log.make_message("user", file_text).tokens 
                    if tokens > self.chat_wrapper.chat_log.max_chat_tokens:
                        print(
                            ms.red(f"Warning: Your input is {tokens - self.chat_wrapper.chat_log.max_chat_tokens} ({str(tokens)} total tokens) tokens too long. The maximum input length for this instance  is {self.chat_wrapper.chat_log.max_chat_tokens} tokens. ")
                        )
                        print("Hint: A token is around 4 characters long. So a 1024 token input will usually have  2/3 as many words.  ")
                        print("You can also modify the max chat tokens by customizing a template. Head over to the template documentation in the docs folder for more information.")
                        print("For more information, head over to this link:")
                        print("https://platform.openai.com/tokenizer")
                        print("File could not be loaded. Please try again.")
                        continue 
                    print("File loaded successfully!")
                    print("> " + file_text)
                    print("Generating response...")
                    response = self.chat_wrapper.chat_with_assistant(file_text)
                    print(response)

            elif ans_lower in (
                "clear",
                "r",
                "reset",
            ):
                if confirm(
                    "Are you sure you want to clear the chat log? You cannot undo this. "
                ):
                    self.chat_log.reset()
                    print("Chat log cleared.")
                else:
                    print("Chat log not cleared.")
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("c", "chunk"):
                print(
                    "Hint: Chunking requires you to press enter twice to submit your input. In many terminals, there is a max input length of 1024 characters, so chunking is useful for longer inputs."
                )
                print("Toggling chunking")
                self.chunking = toggle(self.chunking)
                print(f"Chunking is now {bool_to_yes(self.chunking, y= 'on', n='off')}")
            elif ans_lower in ("load", "l"):
                print("Loading menu...")
                self.load_menu()
                if self.chat_wrapper.is_loaded:
                    print(self.chat_wrapper.chat_log.get_pretty_messages())
            elif ans_lower in ("params", "p"):
                print("Entering the model parameter menu...")
                self.modify_model_param_menu()
                print("Returning to the chat loop...")
            elif ans_lower == "debug":
                print("Printing debug info...")
                print(self.chat_wrapper.__repr__())
            elif ans_lower in ("print", "pr"):
                print(self.chat_wrapper.chat_log.get_pretty_messages())

            elif ans_lower in ("quicksys", "qsys"):
                print("Entering the quick system prompt menu...")
                new_prompt = self.sys_manager_obj.quick_sys_prompt(save=False)
                if new_prompt is None:
                    print("No system prompt loaded.")
                    print("Returning to the chat loop...")
                    continue
                self.chat_wrapper.chat_log.sys_prompt = new_prompt
                print("System prompt changed.")
                print("Returning to the chat loop...")
                continue
            elif ans_lower in ("sysmanage", "sysm", "sm"):
                print("Entering the system prompt manager menu...")
                self.sys_manager_obj.main_menu()
                if self.sys_manager_obj.loaded_file is not None:
                    self.chat_wrapper.chat_log.sys_prompt = (
                        self.sys_manager_obj.loaded_file
                    )
                    print("System prompt changed.")
                    print("Returning to the chat loop...")
                    continue
                else:
                    print("No system prompt loaded.")
                    print("Returning to the chat loop...")
                    continue
            elif ans_lower in ("sys", "system"):
                new_prompt = self.sys_manager_obj.quick_sys_prompt(save=True)
                if new_prompt is None:
                    print("No system prompt loaded.")
                    print("Returning to the chat loop...")
                    continue
                else:
                    self.chat_wrapper.chat_log.sys_prompt = new_prompt
                    if self.chat_wrapper.chat_log._sys_prompt == new_prompt:
                        print("System prompt changed.")
                    else:
                        print(
                            "A error has occurred that indicates a deep problem with this code base. Please report this issue on the github page, along with the following information:"
                        )
                        print(repr(self.chat_wrapper))

            else:
                if ans == "" or ans == " " or ans == None:
                    print("No input detected.")
                    continue
                response = self.chat_wrapper.chat_with_assistant(ans)
                print(response)

    def load_menu(self):
        """A menu for loading chat logs"""
        msg_list = [
            "Type a filename to load a chat log",
            "As per usual, type 'quit' to quit(and return to main menu)",
            "Type help to see this message again",
            "Type list to see a list of files in the current directory",
        ]
        file_list = self.chat_wrapper.save_and_load.get_files(remove_path=True)
        files = "\n".join(
            ["The following is a list of currently saved_files: "] + file_list
        )
        message = "\n".join(msg_list)
        print(message)
        print(files)
        while True:
            ans = input("> ")
            ans_lower = ans.lower()
            if ans_lower in ("quit", "q"):
                if confirm("Are you sure you want to quit?"):
                    print("Returning to previous menu...")
                    break
                else:
                    print("Not quitting. Type 'help' to see the help message again.")
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("list", "l"):
                print(files)
            else:
                if ans in file_list:
                    print(f"Loading {ans}...")
                    if confirm(
                        "Are you sure you want to load from this file? Any previous chat will be overwritten."
                    ):
                        try:
                            if self.chat_wrapper.load(ans):
                                print("Chat loaded. Returning to previous menu...")
                                return True
                            else:
                                print(
                                    "Selected file was not found. If you believe this is an error, please contact the developer."
                                )
                                continue
                        except Exception as e:
                            print(
                                "An error occurred while loading the chat. Please try again."
                            )
                            print(e)
                    else:
                        print("Chat not loaded.")
                        print("Returning to previous menu...")
                        continue
                else:
                    print("File not found. Please try again.")

    def save_menu(self) -> None:
        """Save menu for chat logs"""
        msg_list = [
            "Type a filename for the chat log save file",
            "Type 'quit' to quit(and return to main menu)",
            "If the file already exists, you will be asked to confirm overwriting it"
            "Type help to see this message again",
            "Type enter to save chat log with a timestamp as the filename",
            "The following is a list of files in the current directory:",
        ]
        file_list = self.chat_wrapper.save_and_load.get_files(remove_path=True)
        msg_list.extend(file_list)
        msg = "\n".join(msg_list)
        print(msg)
        while True:
            ans = input("> ")
            if ans.lower() in ("quit", "q"):
                if confirm("Are you sure you want to quit?"):
                    break
            elif ans.lower() in ("help", "h"):
                print(msg)
            else:
                if self.chat_wrapper.save(file_name=ans):
                    print("Save successful!")
                    print("Returning to main menu...")
                    break
                else:
                    if confirm(
                        f"File: {ans} already exists. Are you sure you want to overwrite it?"
                    ):
                        if self.chat_wrapper.save(file_name=ans, overwrite=True):
                            print("Save successful!")
                            print("Returning to main menu...")
                            break
                        else:
                            print("Save failed. Please try again.")
                            continue
                    else:
                        print("Save failed. Please try again.")
                        continue

    def modify_model_param_menu(self) -> None:
        """A menu for modifying model parameters"""
        msg_list = [
            "Welcome to the model parameter modification menu!",
            "Explore different parameters to impact the chatbot's responses.",
            "'quit': Exit and return to main menu.",
            "'help': Display this help message again.",
            "'default': Reset all parameters to default values.",
            "'list': Display a list of all parameters.",
            "'set {parameter_name}': Set a specific parameter. Must use the exact parameter name.",
            "'param_help {parameter_name}': Get a description of a specific parameter. Use without a parameter name for a list of all parameters and their descriptions.",
        ]

        def make_param_list():
            return [
                f"{key}: {value}"
                for key, value in self.chat_wrapper.gpt_chat.get_params().items()
            ]

        msg = "\n".join(msg_list)
        current_params = "\n".join(make_param_list())
        print(msg)
        print("Current parameters:")
        print(current_params)
        possible_params = self.gpt_chat.possible_optional_params
        param_help_dict = self.gpt_chat.param_help

        while True:
            ans = input("> ")
            ans_lower = ans.lower()
            if ans_lower in ("q", "quit", "exit"):
                print("Returning to main menu...")
                break
            elif ans_lower in ("help", "h"):
                print(msg)
            elif ans_lower in ("list", "l"):
                print("Possible parameters:", " ".join(list(possible_params)))
                print("Current parameters:")
                print("\n".join(make_param_list()))
            elif ans_lower.startswith("param_help"):
                param = split_input_from_cmd("param_help", ans_lower)
                if param is None or param == "" or param not in possible_params:
                    print("Possible parameters:", " ".join(list(possible_params)))
                    print("Descriptions:")
                    print(format_dict_as_string(param_help_dict))
                else:
                    print(param_help_dict[param])

            elif ans_lower.startswith("set"):
                param = split_input_from_cmd("set", ans_lower)
                param = param.strip()
                if param is None or param not in possible_params:
                    print("Invalid parameter. Please try again.")
                    print("Possible parameters:", " ".join(list(possible_params)))
                else:
                    print(
                        "Please note, in order to set a parameter to a value less than one you must begin the value with a 0."
                    )
                    print("For example, '.1' will not work, but '0.1' will.")
                    print(
                        "To remove a value entirely(and use OpenAI's default), type 'None' (without quotes)"
                    )
                    print("To set a value to zero type 'zero' (without quotes)")
                    value = input(f"Enter a value for {param}: ")

                    try:
                        self.gpt_chat.modify_params(**{param: value})
                    except fact.cw.g.BadChatCompletionParams as e:
                        print("Invalid Parameter. More info:")
                        print(e)
                        continue
                    print("Parameter set successfully!")
                    current_params_list = [
                        f"{key}: {value}"
                        for key, value in self.chat_wrapper.gpt_chat.get_params().items()
                    ]
                    current_params = "\n".join(current_params_list)
                    print("Current parameters:")
                    print(current_params)

            elif ans_lower == "default":
                if confirm("Are you sure you want to reset all parameters?"):
                    self.gpt_chat.reload_from_template()
                    print("Parameters reset successfully!")
                    current_params_list = [
                        f"{key}: {value}"
                        for key, value in self.chat_wrapper.gpt_chat.get_params().items()
                    ]
                    current_params = "\n".join(current_params_list)
                    print("Current parameters:")
                    print(current_params)

            else:
                print("Invalid command. Please try again.")


class MainMenu:
    def __init__(
        self,
        sys_prompt_manager: SysPromptManagerMenu,
        API_KEY=API_KEY,
        factory: fact.ChatWrapperFactory = fact.wrapper_factory,
    ):
        self.API_KEY = API_KEY
        self.factory: fact.ChatWrapperFactory = factory
        self.selected_template_name = DEFAULT_TEMPLATE_NAME
        try:
            self.factory.select_template(self.selected_template_name)
        except self.factory.template_selector.TemplateNotFoundError:
            self.factory.select_template(get_default_template_for_model(DEFAULT_MODEL))
        self.is_default_template = True
        self.chat_wrapper: fact.cw.g.GPTChat = None
        self.is_ready = False
        self.template_selector = self.factory.template_selector
        self.sys_prompt_manager = sys_prompt_manager
        self.system_prompt = "You are a helpful AI assistant. Your model is {model}. Today's date is {date}. Your training data was last updated in September 2021."

    def _get_template_info(self, template_name: str) -> str:
        info = self.template_selector.get_template_info(template_name)
        for key, value in info.items():
            print(f"{key}: {value}")

    def template_menu(self) -> None:
        def format_template_dict_as_string(name: str, template_dict: dict) -> str:
            result = []
            result.append(f"Template Name: {name}")
            result.append("Description: " + template_dict["description"])
            result.append("Tags: " + ", ".join(template_dict["tags"]))
            result.append("ChatLog  Parameters:")

            for key, value in template_dict["chat_log"].items():
                result.append(f"{key.title()}: {value}")
            result.append("Completion Parameters:")
            for key, value in template_dict["gpt_chat"].items():
                result.append(f"{key.title()}: {value}")
            return "\n".join(result)

        msg_list = [
            "Welcome to the template menu!",
            "Type 'quit' to quit(and return to main menu)",
            "Type 'help' to see this message again",
            "Type list to see a list of all templates",
            "Type 'set {template_name}' to set the template",
            "Type 'info {template_name}' to see info about a specific template",
            "Type 'current' to see the current template",
        ]
        current_temp_info_list = [
            "Current Selected Template Name: " + self.selected_template_name,
            "Is Default Template: " + bool_to_yes(self.is_default_template),
            "Current Template Info:",
            format_dict_as_string(
                self.template_selector.get_template_info(self.selected_template_name)
            ),
        ]
        all_templates_dict = self.template_selector.get_all_templates()

        all_template_info = [
            format_template_dict_as_string(name, template_dict)
            for name, template_dict in all_templates_dict.items()
        ]
        message = "\n".join(msg_list)

        current_temp_info = "\n".join(current_temp_info_list)

        available_templates = ", ".join(list(all_templates_dict.keys()))
        print(ms.magenta("===( Current Template Info )==="))
        print(current_temp_info)
        print("---" * 10)
        print(ms.magenta("===( Template Menu )==="))

        print(message)

        print("Available Templates: " + available_templates)
        while True:
            ans = input("> ")
            ans_lower = ans.lower()
            if ans_lower in ("q", "quit", "exit"):
                print("Returning to main menu...")
                break
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("list", "l"):
                print("Available Templates: " + available_templates)
            elif ans_lower.startswith("info"):
                cmd = split_input_from_cmd("info", ans_lower)
                if cmd is None:
                    print("Printing info for all templates...")
                    print(all_template_info)
                elif cmd.strip() in all_templates_dict.keys():
                    print(format_template_dict_as_string(cmd, all_templates_dict[cmd]))
                else:
                    print("Invalid template name. Please try again.")
                    print("Available Templates: " + available_templates)
                    continue
            elif ans_lower.startswith("set"):
                cmd = split_input_from_cmd("set", ans_lower)
                if cmd is None:
                    print("Invalid template name. Please try again.")
                    print("Available Templates: " + available_templates)
                    continue
                elif cmd not in all_templates_dict.keys():
                    print("Invalid template name. Please try again.")
                    print("Available Templates: " + available_templates)
                    continue
                else:
                    if confirm(
                        "Are you sure you want to set the template to " + cmd + "?"
                    ):
                        self.selected_template_name = cmd
                        self.factory.select_template(self.selected_template_name)
                        self.is_default_template = False
                        print("Template set successfully!")
                        print("Current Template Info:")
                        print(
                            format_dict_as_string(
                                self.template_selector.get_template_info(
                                    self.selected_template_name
                                )
                            )
                        )
                        print(
                            "Is Default Template: "
                            + bool_to_yes(self.is_default_template)
                        )
                    else:
                        print("Template not set.")
                        print(
                            "Type 'set {template_name}' to set the template, or 'quit' to quit."
                        )
                        continue
            elif ans_lower.startswith("current"):
                print("Current Selected Template Name: " + self.selected_template_name)
                print("Is Default Template: " + bool_to_yes(self.is_default_template))
                print("Current Template Info:")
                print(
                    format_dict_as_string(
                        self.template_selector.get_template_info(
                            self.selected_template_name
                        )
                    )
                )
            else:
                print("Invalid command. Please try again.")
                continue

    def sys_prompt_menu(self) -> None:
        msg_list = [
            "To quickly type up a new system prompt, without the option to save it type 'new' ",
            "To do the same but be prompted to save it, type 'new save'",
            "To enter the full system prompt manager, type 'full'",
            "To quickly enter the system prompt loading menu, type 'load'",
            "To exit and return to the main menu, type 'quit'",
            "To see this message again, type 'help'",
        ]
        print("Welcome to the system prompt manager!")
        message = "\n".join(msg_list)
        print(message)
        while True:
            if self.system_prompt is not None:
                print("Current System Prompt: " + self.system_prompt)
                print(
                    "Keep in mind, that {wildcards} will be replaced with the appropriate values. To see a full list of wildcards, type 'wildcards'"
                )
            ans = input("> ")
            ans_lower = ans.lower()
            ans_lower = ans_lower.strip()
            if ans_lower in ("quit", "q"):
                print("Returning to main menu...")
                return None
            elif ans_lower in ("load", "l"):
                print("Entering system prompt loading menu...")
                new_sys_prompt = self.sys_prompt_manager.load_menu()
                if new_sys_prompt is not None:
                    self.system_prompt = new_sys_prompt
                    print("System prompt set to: " + self.system_prompt)
                    print("Exiting system prompt menu and returning to main menu...")
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("wildcards", "wc"):
                print("Currently Available Wildcards: ")
                print(self.sys_prompt_manager.get_wildcards())
            elif ans_lower in ("new", "n"):
                new_prompt = self.sys_prompt_manager.quick_sys_prompt(save=False)
                if new_prompt is None:
                    print("System prompt not saved.")
                    continue
                else:
                    self.system_prompt = new_prompt
                    print("System prompt set to: " + self.system_prompt)
                    print("Returning to main menu...")
                    break
            elif ans_lower in ("new save", "ns"):
                print("Entering quick system prompt menu...")
                new_prompt = self.sys_prompt_manager.quick_sys_prompt(save=True)
                if new_prompt is None:
                    print("System prompt not saved.")
                    print("Still in the main menu's system prompt menu...")
                    continue
                else:
                    self.system_prompt = new_prompt
                    print(
                        "System prompt set to: " + self.system_prompt,
                    )
                    print("Returning to main menu...")
                    break
            elif ans_lower in ("full", "f"):
                self.sys_prompt_manager.main_menu()
                if self.sys_prompt_manager.loaded_file is not None:
                    print(
                        "System prompt set to: " + self.sys_prompt_manager.loaded_file
                    )
                    self.system_prompt = self.sys_prompt_manager.loaded_file
                    print("Returning to main menu...")
                    break
                else:
                    print("Still in the main menu's system prompt menu...")
            else:
                print("Invalid command. Please try again.")
                continue

    def load_menu(self) -> None:
        self.chat_wrapper = self.factory.make_chat_wrapper()
        # yes, I did repeat myself. But it might be useful for users to be able to load from a save in both the main menu and the chat menu
        msg_list = [
            "Type a filename to load a chat log",
            "As per usual, type 'quit' to quit(and return to main menu)",
            "Type help to see this message again",
            "Type list to see a list of files in the current directory",
        ]
        file_list = self.chat_wrapper.save_and_load.get_files(remove_path=True)
        files = "\n".join(
            ["The following is a list of currently saved_files: "] + file_list
        )
        message = "\n".join(msg_list)
        print(message)
        print(files)
        while True:
            ans = input("> ")
            ans_lower = ans.lower()
            if ans_lower in ("quit", "q"):
                if confirm("Are you sure you want to quit?"):
                    print("Returning to previous menu...")
                    break
                else:
                    print("Not quitting. Type 'help' to see the help message again.")
            elif ans_lower in ("help", "h"):
                print(message)
            elif ans_lower in ("list", "l"):
                print(files)
            else:
                if ans in file_list:
                    print(f"Loading {ans}...")
                    if confirm(
                        "Are you sure you want to load a save? Any previously selected templates or system prompts will be discarded."
                    ):
                        try:
                            if self.chat_wrapper.load(ans):
                                print("Chat loaded.")
                                if confirm(
                                    "Would you like to start chatting immediately?"
                                ):
                                    self.start_chat()
                                return True
                            else:
                                print(
                                    "Selected file was not found. If you believe this is an error, please contact the developer."
                                )
                                continue
                        except Exception as e:
                            print(
                                "An error occurred while loading the chat. Please try again."
                            )
                            print(e)
                    else:
                        print("Chat not loaded.")
                        print("Returning to previous menu...")
                        continue
                else:
                    print("File not found. Please try again.")

    def _make_chat_wrapper(self):
        self.factory.select_template(self.selected_template_name)
        self.chat_wrapper = self.factory.make_chat_wrapper()
        self.chat_wrapper.chat_log.sys_prompt = self.system_prompt
        self.chat_loop = ChatLoop(
            sys_manager_obj=self.sys_prompt_manager, chat_wrapper=self.chat_wrapper
        )
        self.is_ready = True

    def start_chat(self) -> None:
        if self.is_ready:
            self.chat_loop.run_chat_loop()
            print("Chat loop exited.")
            print("Returning to main menu...")
            self.chat_loop = None
            return None
        else:
            try:
                if self.chat_wrapper is None:
                    self._make_chat_wrapper()
                else:
                    self.chat_loop = ChatLoop(
                        sys_manager_obj=self.sys_prompt_manager,
                        chat_wrapper=self.chat_wrapper,
                    )

                self.chat_loop.run_chat_loop()
                self.chat_loop = None
                print("Returning to main menu...")
            except Exception as e:
                print(e)
                print(
                    "Error when setting up chat bot. Please make sure you have a valid template selected and a valid system prompt."
                )

    def main_menu(self) -> None:
        if BYPASS_MAIN_MENU == True:
            print("Bypass main menu is set to true. ")
            print(
                "You can turn this off by setting BYPASS_MAIN_MENU to 0 in the .env file."
            )
            print("Starting chat...")
            self.start_chat()
            print("Exiting program...")
            return None
        print(ms.magenta("===================================="))
        print(
            ms.yellow(
                "Welcome to the main menu for Alex's Maybe Kinda Neat CLI Chat Interface!"
            )
        )
        print(
            "Note: Ensure that you have completed the setup process before attempting to use the chat bot, otherwise you will encounter errors. For help with that, please look over the README.md file, or better yet HELP_ME.md file in the docs folder."
        )
        print(ms.magenta("===================================="))
        msg_list = [
            "Please note, you can bypass the main menu entirely by setting the .env variable 'BYPASS_MAIN_MENU' to 1.",
            "Type 'chat' to start a chat",
            "Type 'template' to enter the template menu",
            "Type 'system' to enter the system prompt menu",
            "Type 'load' to load from a previous save.",
            "Type 'help' to see this message again.",
            "Type 'quit' to quit. (This will exit the program entirely)",
        ]
        CmdAndInfo = namedtuple(
            "CmdAndInfo", ["func", "alt_cmds", "name", "start_msg", "end_msg"]
        )
        map_dict = {
            "chat": CmdAndInfo(
                self.start_chat,
                ("c", "go"),
                "chat",
                "Entering chat menu...",
                "Returning to main menu...",
            ),
            "template": CmdAndInfo(
                self.template_menu,
                ("t", "temp"),
                "template",
                "Entering template menu...",
                "Back in main menu...",
            ),
            "load": CmdAndInfo(
                self.load_menu,
                ("l", "load"),
                "load",
                "Entering load menu...",
                "Back in main menu...",
            ),
            "system": CmdAndInfo(
                self.sys_prompt_menu,
                ("s", "sys"),
                "system",
                "Entering system prompt menu...",
                "Back in main menu...",
            ),
        }

        def process_command(user_input):
            for cmd, cmd_info in map_dict.items():
                if user_input == cmd or user_input in cmd_info.alt_cmds:
                    print(cmd_info.start_msg)
                    cmd_info.func()  # Call the function associated with the command
                    print(cmd_info.end_msg)
                    return True
            return False

        message = "\n".join(msg_list)
        print(message)

        while True:
            print("On main menu. Please enter a command.")
            ans = input("> ")
            ans_lower = ans.lower()
            ans_lower = ans_lower.strip()

            if ans_lower in ("q", "quit", "exit"):
                if confirm("Are you sure you want to quit?"):
                    print("Quitting...")
                    break
                else:
                    print("Not quitting. Type 'help' to see the help message again.")
            elif ans_lower in ("h", "help"):
                print(message)
            elif ms.xy(ans_lower):
                pass

            else:
                if not process_command(ans_lower):
                    print("Invalid command. Please try again.")
                    continue
                else:
                    continue


sys_prompt_manager = SystemPromptManager()
sys_menu = SysPromptManagerMenu(sys_prompt_manager)
main_menu = MainMenu(sys_menu)
