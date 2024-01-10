import datetime
import json
import os
import unittest
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import openai

import GPTchat as g
from settings import API_KEY


class BadReturnTypeError(Exception):
    def __init__(
        self,
        message: str = None,
        return_type: str = None,
        possible_return_types: set = None,
    ):
        message_list = []
        if message is None:
            msg_ini = "Incorrect return type"
        else:
            msg_ini = message
        message_list.append(msg_ini)
        if not return_type is None:
            message_list.append(
                f"return type: {return_type} is not a valid return type"
            )
        if possible_return_types is None:
            possible_return_types = {
                "string",
                "dict",
                "Message",
                "pretty_printed",
                "pretty",
            }
        else:
            possible_return_types = set(possible_return_types)
        message_list.append(f"possible return types are {str(possible_return_types)}")
        self.message = "\n".join(message_list)

    def __str__(self):
        return self.message


class ChatWrapperNotSetupError(Exception):
    def __init__(self, message: str = None):
        if message is None:
            message = "ChatWrapper not setup"
        super().__init__(message)


class ChatWrapper:
    version = "1.0.1"

    # Should have an API_KEY set to a variable called API_KEY if you don't want to provide one during construction, (Need one to load the GPTChat object)
    def __init__(
        self,
        API_KEY: str = API_KEY,
        gpt_chat: g.GPTChat = None,
        chat_log: g.ch.ChatLog = None,
        save_path: str = "chatbot_saves",
        return_type: str = "Message",
        wrapper_return_type: str = "pretty_printed",
        default_system_prompt: str = "You are a helpful AI assistant. Your model is {model} Today's date is {date}, and your training data cuts off in September 2021 "
    ) -> None:
        self.constructor_args = {
            "API_KEY": "Excluded for security reasons",
            "gpt_chat": "GPTChat" if not gpt_chat is None else None,
            "chat_log": "ChatLog" if not chat_log is None else None,
            "save_path": save_path,
            "return_type": return_type,
            "wrapper_return_type": wrapper_return_type,
        }
        self.gpt_chat = gpt_chat
        self.chat_log = chat_log
       
        if not self.gpt_chat is None:
            self.gpt_chat.return_type = "Message"
        
            
        self.uuid = str(uuid.uuid4())
        self.is_loaded = False
        self.save_and_load = self.SaveAndLoad(self, save_folder=save_path)
        if self.chat_log is None or self.gpt_chat is None:
            self.is_setup = False
        else:
            self.is_setup = True
        self._wrapper_return_type = self._check_return_type(wrapper_return_type)
        self.API_KEY = API_KEY
        if not self.chat_log is None:
            self.chat_log.sys_prompt = default_system_prompt

    possible_return_types = {"string", "dict", "Message", "pretty_printed", "pretty"}

    def _check_return_type(self, return_type: str) -> str:
        """Checks if return type is in possible return types, raises BadReturnTypeError if not"""
        if not return_type in self.possible_return_types:
            raise BadReturnTypeError(
                return_type=return_type,
                possible_return_types=self.possible_return_types,
            )
        return return_type
    def set_sys_prompt(self, sys_prompt: str) -> None:
        self.chat_log.sys_prompt = sys_prompt
    def _format_return_type(self, message: g.ch.Message) -> str:
        """Takes Message object and returns it in the format specified by the wrapper_return_type attribute"""
        if self.wrapper_return_type == "dict":
            return dict(message)
        elif self.wrapper_return_type == "Message":
            return message
        elif (
            self.wrapper_return_type == "pretty_printed"
            or self.wrapper_return_type == "pretty"
        ):
            return message.pretty()
        elif self.wrapper_return_type == "string":
            return message.content
        else:
            return message

    @property
    def wrapper_return_type(self) -> str:
        return self._wrapper_return_type

    @wrapper_return_type.setter
    def wrapper_return_type(self, return_type: str) -> None:
        """Sets the return type of the wrapper, must be one of the following: string, dict, Message, pretty_printed"""
        self._wrapper_return_type = self._check_return_type(return_type)

    def _check_setup(self) -> None:
        """Raises a type error if the chatbot has not been setup"""
        if self.chat_log is not None and self.gpt_chat is not None:
            self.is_setup = True
        if not self.is_setup:
            raise ChatWrapperNotSetupError(
                "Chatbot has not been setup, please provide a GPTChat object and a ChatLog object to the constructor"
            )

    def run_chat(self) -> None:
        """Sends chatlog to API and adds response to chatlog, uses the GPTChat object's make_api_call method. If an error occurs while making an API call, the chatbot will be saved to a file with the current time as the name and the error will be raised"""
        self._check_setup()
        self.gpt_chat.return_type = "string"
        try:
            response = self.gpt_chat.make_api_call(
                self.chat_log.get_finished_chat_log()
            )
            self.chat_log.assistant_message = response
        except openai.OpenAIError as e:
            print("A fatal error occurred while making an API call to OpenAI's API")
            save_name = (
                datetime.datetime.now().isoformat().replace(":", "-") + "_fatal_error"
            )
            print("Saving chatbot to " + save_name + " before exiting...")
            if self.save(save_name):
                print("Chatbot saved successfully")
            raise e

       
    def get_string_from_response(self, response: g.ch.Message | dict | str ) -> str:
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response["content"]
        elif isinstance(response, g.ch.Message):
            return response.content
    def add_GPTChat_object(self, gpt_chat: g.GPTChat) -> None:
        """Adds a GPTChat object to the chatbot"""
        if not isinstance(gpt_chat, g.GPTChat):
            raise TypeError(
                "gpt_chat must be an instance of GPTChat, not " + str(type(gpt_chat))
            )

        self.gpt_chat: g.GPTChat = gpt_chat
        self.API_KEY = gpt_chat.api_key
        self.gpt_chat.return_type = "string"

    def add_ChatLog_object(self, chat_log: g.ch.ChatLog) -> None:
        """Adds a ChatLog object to the chatbot"""
        if not isinstance(chat_log, g.ch.ChatLog):
            raise TypeError(
                "chat_log must be an instance of ChatLog, not " + str(type(chat_log))
            )
        self.chat_log: g.ch.ChatLog = chat_log

    @property
    def assistant_message(self) -> str:
        """Returns the assistant message"""
        self._check_setup()

        return self.chat_log.assistant_message

    @assistant_message.setter
    def assistant_message(self, message: str) -> None:
        """Sets the assistant message"""
        self._check_setup()
        self.chat_log.assistant_message = message

    @property
    def user_message(self) -> str:
        """Returns the user message, pretty printed"""
        return self._format_return_type(self.chat_log.user_message)

    @user_message.setter
    def user_message(self, message: str) -> None:
        """Sets the user message"""
        self.chat_log.user_message = message

    def chat_with_assistant(self, message: str) -> str:
        """Sets an assistant message and returns the response, pretty printed"""
        self._check_setup()
        if message == "" or None:
            message = "  "
        self.user_message = message
        self.run_chat()
        return self._format_return_type(self.assistant_message)

    def save(self, file_name: str, overwrite: bool = False) -> bool:
        """Wrapper for the save_and_load object's save_to_file method"""
        return self.save_and_load.save_to_file(file_name, overwrite)

    def load(self, file_name: str) -> bool:
        """Wrapper for the save_and_load object's load_from_file method"""
        return self.save_and_load.load_from_file(file_name)

    def modify_max_completion_tokens(self, max_completion_tokens: int) -> None:
        """This is necessary as the max_completion_tokens must be changed in both the ChatLog object and the GPTChat object"""
        self.chat_log.set_token_info(max_completion_tokens=max_completion_tokens)
        self.gpt_chat.max_tokens = max_completion_tokens
    def __repr__(self):
        msg_list = []
        constructor_arg_list = []
        for key, val in self.constructor_args.items():
            constructor_arg_list.append(key + "=" + str(val))
        msg_list.append("ChatWrapper(" + ", ".join(constructor_arg_list) + ")")
        msg_list.append("ChatWrapper object with the following attributes:")
       
        msg_list.append("    is_setup: " + str(self.is_setup))
        msg_list.append("    wrapper_return_type: " + self._wrapper_return_type)
        msg_list.append("    is_loaded: " + str(self.is_loaded))
        msg_list.append("    Version: " + str(self.version))
        msg_list.append("    Save and Load object: " + repr(self.save_and_load)) 
        msg_list.append("--------------------")
        msg_list.append("ChatLog Object Information:")
        msg_list.append(self.chat_log.__repr__())
        msg_list.append("--------------------")
        msg_list.append("GPTChat Object Information:")
        msg_list.append(self.gpt_chat.__repr__())
        
        return "\n".join(msg_list)
        
    class SaveAndLoad:
        """
        Class for saving and loading chat wrappers, with separate methods for saving and loading to file, and saving and loading to dictionary"
        Methods:
            save_to_file(file_name: str, overwrite: bool = False) -> bool
            load_from_file(file_name: str) -> bool
            get_files(remove_path = True) -> list
            make_save_dict(file_name: str) -> dict
            load_from_dict(save_dict: dict) -> None
            _verify_save_dict(save_dict: dict) -> dict
        Attributes:
            chat_wrapper: The chat wrapper to save or load from
            save_folder: The folder to save to or load from
            gpt_chat: The GPTChat object inside the chat wrapper
            chat_log: The ChatLog object inside the chat wrapper
        Example Usage:
            chat_wrapper = ChatWrapper(gpt_chat, chat_log)
            chat_wrapper.save_and_load.save_to_file("chat_wrapper_save")
            chat_wrapper.save_and_load.load_from_file("chat_wrapper_save")
            chat_wrapper.save_and_load.get_files()
        """

        def __init__(self, chat_wrapper, save_folder="chat_wrapper_saves"):
            self.chat_wrapper = chat_wrapper
            if not save_folder.endswith("/"):
                save_folder = save_folder + "/"
            self.save_folder = save_folder
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
            self.gpt_chat = chat_wrapper.gpt_chat
            self.chat_log = chat_wrapper.chat_log
        def __repr__(self):
            return "SaveAndLoad(chat_wrapper, save_folder=" + repr(self.save_folder) + ")"

        def make_save_dict(self) -> dict:
            """Returns a dictionary that can be used to recreate the chat wrapper"""

            chat_log_dict = self.chat_log.make_save_dict()

            gpt_chat_dict = self.gpt_chat.make_save_dict()
            gpt_chat_dict['return_type'] = "Message"

            meta_data = {
                "chat_wrapper_version": self.chat_wrapper.version,
                "chat_wrapper_uuid": self.chat_wrapper.uuid,
                "timestamp": datetime.datetime.now().isoformat().replace(":", "-"),
            }
            return {
                "meta_data": meta_data,
                "chat_log": chat_log_dict,
                "gpt_chat": gpt_chat_dict,
            }

        def load_save_dict(self, save_dict: dict, API_KEY: str = None) -> None:
            """Loads a save dict into the chat wrapper"""
            self.chat_wrapper.uuid = save_dict["meta_data"]["chat_wrapper_uuid"]
            if API_KEY is None:
                API_KEY = self.chat_wrapper.API_KEY
            self.chat_log = g.ch.ChatLog()
            self.chat_wrapper.add_ChatLog_object(self.chat_log)
            self.chat_wrapper.chat_log.save_to_dict.load(save_dict["chat_log"])
            
            self.gpt_chat = g.GPTChat(API_KEY=API_KEY, return_type="Message")
            self.gpt_chat.load_save_dict(save_dict["gpt_chat"])
            self.chat_wrapper.add_GPTChat_object(self.gpt_chat)
            self.chat_wrapper.is_loaded = True
            self.chat_wrapper.is_setup = True

        def save_to_file(self, file_name: str, overwrite=False) -> bool:
            """Saves the chat wrapper to a file, returns True if successful, False if not"""
            if not overwrite and os.path.exists(file_name):
                return False
            file_name = self._add_file_path(file_name)
            save_dict = self.make_save_dict()
            with open(file_name, "w") as f:
                json.dump(save_dict, f)
            return True

        def load_from_file(self, file_name: str) -> None:
            """Loads a save file into the chat wrapper"""
            file_name = self._add_file_path(file_name)
            try:
                with open(file_name, "r") as f:
                    save_dict = json.load(f)
                self.load_save_dict(save_dict)
            except FileNotFoundError:
                print("File not found: " + file_name)
                return False
            return True

        def _add_file_path(self, file_name: str) -> str:
            """Adds the path to the file name, as well as the .json extension"""

            if not file_name.endswith(".json"):
                file_name = file_name + ".json"
            if not file_name.startswith(self.save_folder):
                file_name = self.save_folder + file_name
            return file_name

        def _remove_file_path(self, file_name: str) -> str:
            """Removes the path from the file name, as well as the .json extension"""
            if file_name.startswith(self.save_folder):
                file_name = file_name[len(self.save_folder) :]
            if file_name.endswith(".json"):
                file_name = file_name[:-5]
            return file_name

        def get_files(self, remove_path=True) -> list:
            """Returns a list of files in the save folder, if remove_path is true, the path will be removed from the file names"""
            files = []
            for file in os.listdir(self.save_folder):
                if file.endswith(".json"):
                    if remove_path:
                        file = self._remove_file_path(file)
                    files.append(file)
            return files


class TestChatWrapper(unittest.TestCase):
    def setUp(self):
        self.gpt_chat = g.GPTChat(
            API_KEY=API_KEY,
        )
        self.chat_log = g.ch.ChatLog()
        self.chat_log.sys_prompt = "Hello, how are you?"
        self.chat_wrapper = ChatWrapper(gpt_chat=self.gpt_chat, chat_log=self.chat_log)
        self.maxDiff = None

    def tearDown(self) -> None:
        del self.gpt_chat
        del self.chat_log
        del self.chat_wrapper

    def test_chat_wrapper_basic(self):
        self.assertIsInstance(self.chat_wrapper, ChatWrapper)

    def test_chat_wrapper_save_and_load_basic(self):
        self.assertIsInstance(self.chat_wrapper.save_and_load, ChatWrapper.SaveAndLoad)

    def test_chat_wrapper_save_and_load_make_save_dict(self):
        """
        Test that the save dict is correct, and that it can be loaded back into the chat wrapper

        """
        save_dict = self.chat_wrapper.save_and_load.make_save_dict()
        # must do this because the timestamp is different every time
        save_dict["meta_data"]["timestamp"] = "2021-01-01T00-00-00.000000"

        test_chat_wrapper = ChatWrapper()
        test_chat_wrapper.save_and_load.load_save_dict(save_dict, API_KEY=API_KEY)
        test_chat_wrapper_dict = test_chat_wrapper.save_and_load.make_save_dict()
        # same as above comment
        test_chat_wrapper_dict["meta_data"]["timestamp"] = "2021-01-01T00-00-00.000000"
        self.assertEqual(save_dict, test_chat_wrapper_dict)

    def test_chat_wrapper_save_and_load_save_to_file(self):
        """
        Test that the save file is created, and that it can be loaded back into the chat wrapper

        """
        file_name = "test_chat_wrapper_save_and_load_save_to_file"
        self.chat_wrapper.save_and_load.save_to_file(file_name)
        file_name = self.chat_wrapper.save_and_load._add_file_path(file_name)
        self.assertTrue(os.path.exists(file_name))
        self.chat_wrapper.save_and_load.load_from_file(file_name)
        os.remove(file_name)

    def test_return_type_setter_and_getter(self):
        """Test that the return type can be set and gotten"""
        self.chat_wrapper.wrapper_return_type = "pretty"
        self.assertEqual(self.chat_wrapper.wrapper_return_type, "pretty")

    def test_return_type_error(self):
        """Tests that an error is raised if the return type is not valid"""
        with self.assertRaises(BadReturnTypeError):
            self.chat_wrapper.wrapper_return_type = "not valid"

    def test_setup_check(self):
        """Tests that an error is raised if the chat wrapper is not setup"""
        test_chat_wrapper = ChatWrapper()
        self.assertRaises(ChatWrapperNotSetupError, test_chat_wrapper._check_setup)

    def test_return_type_works(self):
        """Tests that changing the return type will actually change the return type"""
        self.chat_wrapper.chat_log.user_message = "Hello, how are you?"
        self.chat_wrapper.wrapper_return_type = "Message"
        self.assertIsInstance(self.chat_wrapper.user_message, g.ch.Message)


if __name__ == "__main__":
    unittest.main()
