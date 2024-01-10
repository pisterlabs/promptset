import datetime
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import exceptions
import func as f
import openai
import tiktoken
from chat.chatlog import ChatLog
from chat.message import Message, MessageFactory
from chat.system_prompt import SystemPrompt
from chat.trim_chat_log import TrimChatLog
from chat_completion_wrapper import (
    ChatCompletionWrapper,
    ModelParameters,
    ParamInfo,
    param_info,
)
from chat_wrapper.rotate_save import RotatingSave
from handler.save_handler import AbstractCWSaveHandler
from handler.stream_handler import AbstractStreamOutputHandler
from log_config import DEFAULT_LOGGING_LEVEL, BaseLogger
from settings import SETTINGS_BAG


class ChatWrapper:
    version = "3.0.0"
    """
    Updated for v3.0.0
    A wrapper that combines the TrimChatLog and ChatCompletionWrapper classes to make a simple interface for chatbots .
    This is the main facade class for the chat wrapper module, and aside from the handlers and factory, likely the only class you will need to interact with directly. Highly configurable, and can be used in a variety of ways but hopefully not too complex in terms of usage.
    The main functionality is combining the TrimChatLog and ChatCompletionWrapper objects seamlessly, and providing a simple interface for interacting with them, along with providing some extra features like persistence, auto saving, and ensuring that the settings are correctly synced across both objects. 
    Dependencies:
        Custom:
            chat.TrimChatLog -> The object that manages the chat log and trimming it
                chat.ChatLog -> The object that manages the chat log
                chat.SystemPrompt -> The object that manages the system prompt
                chat.MessageFactory -> The object that manages the creation of Message objects
                chat.Message -> The object that represents a message
                AbstractCWSaveHandler -> The object that manages saving the chat log
                AbstractStreamOutputHandler -> The object that manages streaming the output
                RotatingSave -> The object that manages the rotating save system
            chat.ChatCompletionWrapper -> Wrapper that abstracts away the making API calls to the OpenAI API
                chat_completion_wrapper.ModelParameters -> The object that manages the parameters for the model
                chat_completion_wrapper.ParamInfo -> The object that manages the information about the parameters
           
            rotate_save.RotatingSave -> The object that manages the rotating save system
            AbstractCWSaveHandler -> The object that manages saving the chat log
            AbstractStreamOutputHandler -> The object that manages streaming the output
            exceptions -> Custom exceptions
            func -> General functions
            settings(SETTINGS_BAG) -> Settings for the program
            log_config -> Logging configuration (BaseLogger, DEFAULT_LOGGING_LEVEL)
        Installed:
            openai -> The OpenAI API
            tiktoken -> Counts the number of tokens in a string
        Python:
            datetime -> For getting the current time
            time -> For getting a unix timestamp
            logging -> For logging
            typing -> For type hinting
            uuid -> For generating uuids
            json -> used in the attempt_emergency_save method as a final attempt to save the chat log before raising an error
    Raises:
        exceptions.IncorrectObjectTypeError: Raised if an incorrect object type is passed to a method
        exceptions.ObjectNotSetupError: Raised if an object is not set up and an operation is attempted that requires it to be set up
        exceptions.BadRoleError: Raised if an incorrect role is passed to a method
        exceptions.InvalidReturnTypeError: Raised if an invalid return type is passed to a method
        exceptions.MissingValueError: Raised if a required value is missing
    Args:
        API_KEY (str): The OpenAI API key to use
        return_type (str, optional): The return type to use. Defaults to "Message". Must be one of "Message", "str", "dict", "pretty".
        template (dict, optional): An optional template to simplify the setup process. Defaults to None. See templates.py for more info.
        model (str, optional): The model to use. Defaults to None. If None, must be set before chat is called.
        save_handler (AbstractCWSaveHandler, optional): The save handler to use. Defaults to None. If None, must be set before saving is possible.
        is_auto_saving (bool, optional): Whether to enable auto saving. Defaults to SETTINGS_BAG.IS_AUTOSAVING.
    Properties:
        General:
            version (str): The version of the ChatWrapper object
            _model (str): (private) The model to use
            uuid (str): The uuid of the ChatWrapper object, used for logging and debugging
            template (dict): The template to use
            is_trimmed_setup (bool): Whether the TrimChatLog object has been set up
            is_completion_setup (bool): Whether the ChatCompletionWrapper object has been set up
            is_loaded (bool): Whether the ChatWrapper object has been loaded from a save dict
            constructor_args (dict): The arguments used to create the ChatWrapper object(used for __repr__)
        Objects:
            trim_object (TrimChatLog): The TrimChatLog object
            completion_wrapper (ChatCompletionWrapper): The ChatCompletionWrapper object
            message_factory (MessageFactory): The MessageFactory object
            logger (BaseLogger): The logger object for logging(pre-configured logging object )
            rotating_save_handler (RotatingSave): The rotating save handler object
            save_handler (AbstractCWSaveHandler): The save handler object
            stream_handler (AbstractStreamOutputHandler): The stream handler object
            
   Methods:
        Overview:
            1. Core -> The main methods used to interact with the ChatWrapper object
            2. Setup/Template System -> Quick and easy ways to set up the ChatWrapper object's main objects
            3. Making/Managing Objects -> Methods for creating and managing the main objects used by the ChatWrapper object
            4. Special Values that must be the same across both objects -> The following follows are synced across both objects
            5. Change Completion Wrapper Parameters -> Change the parameters of the ChatCompletionWrapper object
            6. Convenience Methods(mostly setters/getters) for Trim Chat Object -> Allows you to set the parameters of the TrimChatLog object and add messages directly to the log without sending anything to the model(While auto-saving if enabled )
            7. Return Type System -> Sets the return type to use
            8. Saving/Loading -> Persistence methods for saving and loading the ChatWrapper object. All aside from the dictionary methods require a save handler to be set up.
            9. Handlers -> Methods for adding handlers to the ChatWrapper object
            10. Auto Save Feature -> Methods used in the auto save feature
            11. Helpers -> Helper methods for the ChatWrapper object, not meant to be used directly
            12. Misc -> Misc methods
            
        1. Core: 
        The main methods used to interact with the ChatWrapper object
            chat(user_message: str | dict | Message) -> str | Message | dict: Sends the given message to the model and returns the response formatted to return type.  
            stream_chat(user_message: str | dict | Message) -> Iterator: Works the same as the chat method, but returns an iterator for special purposes, while processing the messages normally .
            reset() -> None: Resets the trim chat log object but doesn't reset the completion wrapper object. Also deletes all autosaves.
        2. Setup/Template System:
        Quick and easy ways to set up the ChatWrapper object's main objects
            load_template(template: dict = None) -> None: Loads the given template.
            auto_setup_from_template(template: dict = None) -> None: Sets up the ChatWrapper object with the given template. See templates.py for more info on templates.
            reset() -> Reloads chat completion wrapper from template while keeping the trim chat log object the same.
            auto_setup(trim_params: dict = None, completion_params: dict = None) -> None: Sets up the ChatWrapper object with the given parameters. See TrimChatLog and ChatCompletionWrapper for more info on the parameters.
        --------------------
        3. Making/Managing Objects:
        Methods for creating and managing the main objects used by the ChatWrapper object
            make_trim_object(max_tokens: int = 8000, max_completion: int = 1000, system_prompt: str = None, token_padding: int = 500, max_messages: int = 400) -> None: Creates a TrimChatLog object with the given parameters
            make_chat_completion_wrapper(**kwargs) -> None: Creates a ChatCompletionWrapper object with the given parameters
            set_trim_object(trim_object: TrimChatLog) -> None: Sets the TrimChatLog object to the given object
            set_chat_completion_wrapper(completion_wrapper: ChatCompletionWrapper) -> None: Sets the ChatCompletionWrapper object to the given object
            set_trim_token_info(**kwargs) -> None: Sets the token info of the TrimChatLog object to the given parameters
        --------------------
        4. Special Values that must be the same across both objects:
        The following follows are synced across both objects
            system_prompt (str): The system prompt to use
            model (str): The model to use (Setter and getter)
            max_tokens(str) Set the maximum tokens allowed in a completion. This is nessesary as it must be the same as the max_completion_tokens used in the trim object. Otherwise, too many tokens can be included in the finished chat log, which can cause errors.(Setter and getter)
        --------------------
        5. Change Completion Wrapper Parameters:
        Change the parameters of the ChatCompletionWrapper object
            set_chat_completion_params(**kwargs) -> None: Sets the parameters of the ChatCompletionWrapper object to the given parameters
             
            stream-> bool: (getter and setter) Whether to stream the output or not Setter and getter
        --------------------  
        6. Convenience Methods(mostly setters/getters) for Trim Chat Object:
        Allows you to set the parameters of the TrimChatLog object and add messages directly to the log without sending anything to the model(While auto-saving if enabled )
            system_prompt (str): The system prompt to use (Setter and getter)
            reminder (str): The reminder to use (Setter and getter)
            user_message (str): The most recent user message as a string (Getter And Setter )
            assistant_message (str): The most recent assistant message as a string (Getter And Setter )
            get_most_recent_Message(role: str = User, pretty: bool = False) -> Message: Returns the most recent message as a Message object of a given role. Raises an error if the role is not valid. If pretty is True, returns the message as a pretty string.
        --------------------
        7.Return Type System:
        Sets the return type to use
            return_type (str): (getter and setter) The return type to use
        --------------------
        8. Saving/Loading:
        Persistence methods for saving and loading the ChatWrapper object. All aside from the dictionary methods require a save handler to be set up.
            save(save_name: str, overwrite: bool = False) -> None: Saves the ChatWrapper object to the given save name. If overwrite is True, overwrites the save if it already exists. Otherwise, raises an error if the save already exists.
            load(save_name: str) -> None: Loads the ChatWrapper object from the given save name.
            check_entry_name(save_name: str) -> bool: Checks if the given save name exists. Returns True if it does, False if it doesn't.
            delete_entry(save_name: str) -> None: Deletes the given save name.
            all_entry_names() -> List[str]: Returns a list of all the save names.(Getter)
            make_save_dict() -> dict: Makes a save dict from the ChatWrapper object
            load_from_save_dict(save_dict: dict) -> None: Loads the ChatWrapper object from the given save dict
        --------------------   
        9. Handlers:
        Methods for adding handlers to the ChatWrapper object
            add_stream_output_handler(handler: AbstractStreamOutputHandler) -> None: Adds the given stream output handler to the ChatWrapper object
            add_save_handler(save_handler: AbstractCWSaveHandler) -> None: Adds the given save handler to the ChatWrapper object
        --------------------
        9. Auto Save Feature:
        Methods used in the auto save feature
            manual_auto_save() -> None: Manually saves the chat log
            setup_auto_saving() -> None: Sets up the auto saving feature
            setup_autosaving() -> None: Sets up the auto saving feature
            set_auto_save_info(auto_save_frequency: int = None, auto_save_max_saves: int = None, auto_save_entry_name: str = None) -> None: Sets the auto save info to the given parameters
            load_auto_save() -> None: Loads the most recent auto save info from the save handler
            auto_setup_autosaving() -> None: Sets up the auto saving feature
            _auto_save_tick() -> None: (private) Increments the auto save counter and saves the chat log if the counter is greater than the auto save frequency
            manual_auto_save() -> None: Manually saves the chat log(ie creates a new auto save file with the same format as the auto save files)
        --------------------
        11. Helpers:
        Helper methods for the ChatWrapper object, not meant to be used directly
            _check_setup(setup_type: str = None) -> None: (private) Checks if the given setup type has been set up. Raises an error if it has not been set up. (Used for ensuring that required objects are setup before they are used, so a custom specific exception can be raised)
            _process_user_message(user_message: str | dict | Message) -> Message: (private) Processes the given user message into a Message object
            _format_return(response: str) -> str | Message | dict: (private) Formats the response to the given return type
            attempt_emergency_save() -> None: (private) Attempts to save the chat log to a file in the current directory
            _emergency_save() -> None: (private) Saves the chat log to a file in the current directory
            _check_return_type(return_type: str) -> str: (private) Checks if the given return type is valid, returns the return type if it is, otherwise raises an error
            _check_save_handler() -> bool: (private) Checks if a save handler is set up. Returns True if it is, False if it isn't.
            _has_stream_handler() -> bool: (private) Checks if a stream handler is set up. Returns True if it is, False if it isn't.
            _add_stream_to_completion_wrapper() -> None: (private) Adds a stream output handler to the completion wrapper object
            _verify_save_dict(save_dict: dict) -> bool: (private) Verifies that the given save dict is valid. Raises a BadSaveDictError if it is not valid.
        --------------------
        12. Misc:
        
            __repr__() -> str: Returns the repr of the ChatWrapper object
            __str__() -> returns entire chat log pretty formatted as a string
            debug() -> str: Returns the debug information for the ChatWrapper object
            
        
    Example Usage:
    Example 1: Setting up a chat wrapper object and using it to chat with the AI without the chat factory:
        chat_wrapper = ChatWrapper(API_KEY, model="gpt-4")
        chat_wrapper.auto_setup()
        save_handler = JsonSaveHandler()
        stream_handler = StdoutStreamHandler()
        
        chat_wrapper.add_stream_output_handler(stream_handler)
        chat_wrapper.add_save_handler(save_handler)
        chat_wrapper.system_prompt = "This is a test prompt"
        while True:
            user_message = input("You: ")
            if user_message.lower() == "print":
                print(chat_wrapper)
            elif user_message.lower() == "save":
                try:
                    chat_wrapper.save("test_save")
                except exceptions.FileExistsError:
                    print("File already exists")
        
            ...more logic/commands here...
            else:
                chat_wrapper.chat(user_message)
    Example 2:
        This is a more complex example that shows how to set up the chat wrapper object with the chat factory and discord.py. This example also shows how to use the auto saving feature.   We are going to make a chat wrapper, turn off the history feature, set up a save handler and auto-saving, and then use the .stream_chat method to get an iterator for the response, and then send the response to the discord channel as they come in. 
      
        ```
        def make_chat_wrapper() -> ChatWrapper:
            fact = ChatFactory()
            chat_wrapper = fact.get_chat()
            
            # Set up chat wrapper with default parameters
            chat_wrapper.auto_setup()
            
            # Turns off the history feature
            chat_wrapper.trim_object.add_chatlog(None)
            
            # Set system prompt
            chat_wrapper.system_prompt = "This is a test prompt"
            
            # Set save handler
            save_handler = JsonSaveHandler()
            chat_wrapper.add_save_handler(save_handler)
            
            # Set up the auto saving feature with default parameters
            chat_wrapper.auto_setup_autosaving() 

            return chat_wrapper

        # Create the chat wrapper
        chat_wrapper = make_chat_wrapper()

        # Set up discord client here
        @client.event
        async def on_ready():
            print("Ready!")
            
            # Loads the most recent auto save into the chat wrapper object
            chat_wrapper.load_auto_save()

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return 
            
            if message.content.startswith('!chat'):
                # Using the stream_chat method to get an iterator for the response
                gen = chat_wrapper.stream_chat(message.content)
                
                accumulator = ""
                while True:
                    try:
                        token = next(gen)
                        accumulator += token
                        if len(accumulator) >= 1000:
                            await message.channel.send(accumulator)
                            accumulator = ""
                    except StopIteration:
                        # Check if the message is empty
                        if accumulator != "":
                            await message.channel.send(accumulator) # Send the rest of the message
                        break
                ```
        
        Example 3:
        ((Using chat_utils to quickly make a chat wrapper and then test that its working correctly by getting actual responses from the AI))
            import chat_utils as cu
            cw = cu.quick_make_chat_wrapper() # will make a chat wrapper with the default settings
            cu.print_test_ai_response(cw) 
            # gets a response from the AI responding to the message "You are being tested. To confirm you are working correctly, please respond with the message 'All systems are go!'"
               Output:
                 Loading... \.|./.- (Loading spinner)
                 All systems are go! (Loading spinner is no longer visible)
            cu.quick_and_dirty_chatloop(cw) # A simple chat loop that will print the response to the console so we can confirm that its working correctly 
                Output:
                    >>> I am typing this message into the console!
                    (loading spinner)
                    >> Hello, I am an AI assistant! (AI response)
                    >>> So you are working correctly? (User message)
                    ...(loading spinner)...
                    >> Yes, I am working correctly! (AI response)
            
     
     Example 4:
        Using the chat factory to make a chat wrapper object:
        def make_chat_wrapper() -> ChatWrapper:
            fact = ChatFactory()
            fact.select_template("gpt-3-16k_default") # see the template_directory.md in the docs folder for an overview of all built in templates
            return fact.get_chat()
            
        
    """

    def __init__(
        self,
        API_KEY: str,
        return_type: str = "Message",
        model: str = None,
        template: dict = None,
        save_handler: AbstractCWSaveHandler = None,
        is_auto_saving=SETTINGS_BAG.IS_AUTOSAVING,
    ):
        self.constructor_args = {
            "model": model,
            "API_KEY": "exists",
            "return_type": return_type,
        }
        self.logger = BaseLogger(
            __file__, "chat_wrapper.log", "chat_wrapper", level=DEFAULT_LOGGING_LEVEL
        )
        self._model = model
        self.uuid = str(uuid.uuid4())
        self.API_KEY = API_KEY
        self.trim_object = None
        self.completion_wrapper = ChatCompletionWrapper(
            model=self.model, API_KEY=self.API_KEY
        )
        self.template = template
        self.return_type = self._check_return_type(return_type)
        self.message_factory = MessageFactory(model=self.model)
        self.is_trimmed_setup = False
        self.is_completion_setup = False
        self.stream_handler = None
        self.is_loaded = False
        self.save_handler = save_handler

        self.logger.info("Chat Wrapper Created")
        self.logger.debug("Chat Wrapper Created: " + repr(self))
        # auto save system
        self._is_autosaving = bool(is_auto_saving)

        self.rotating_save_handler = RotatingSave(
            self.save_handler
        )  # can deal with save handler being None
        # load in the default values for the rotating save, can be changed with the set_rotating_save_params method
        self._auto_save_frequency = SETTINGS_BAG.AUTO_SAVE_FREQUENCY
        self._auto_save_max_saves = SETTINGS_BAG.AUTO_SAVE_MAX_SAVES
        self._auto_save_entry_name = SETTINGS_BAG.AUTO_SAVE_ENTRY_NAME
        self._auto_save_counter = 0  # used to keep track of how many messages have been sent since the last save(if self._is_autosaving is True)
        self.setup_auto_saving()
        if self._is_autosaving and self._check_save_handler():
            self.logger.info("Auto Saving Enabled")
            self.auto_setup_autosaving()

    # =============(CORE METHODS)================
    """Core methods for the ChatWrapper object. These methods are the main methods used to interact with the ChatWrapper object."""

    def stream_chat(self, user_message: str | dict | Message) -> Iterator:
        """Works the same as the chat method, but returns an iterator for special purposes, while processing the messages normally ."""

        self._check_setup()
        self._auto_save_tick()
        # if streaming isn't on we will set it temporarily on
        curr_streaming = self.completion_wrapper.stream
        self.completion_wrapper.stream = True
        user_message = self._process_user_message(user_message)
        self.trim_object.user_message_as_Message = user_message
        response_object = self.completion_wrapper.stream_chat(
            self.trim_object.get_finished_chatlog()
        )

        response_str = ""
        try:
            for event in response_object:
                stop_reason = event.choices[0].finish_reason
                if stop_reason is not None:
                    # stream has ended
                    # also need to add the message to the chat log
                    msg = self.message_factory(role="assistant", content=response_str)
                    self.trim_object.add_message(msg)
                    return
                else:
                    # yield the token
                    token = event.choices[0].delta.content
                    response_str += token
                    self.logger.debug("Got token: " + token)
                    yield token
        except openai.OpenAIError as e:
            # something went very long, even after 3 retries
            self.logger.critical("OpenAI Error: " + str(e))
            self.completion_wrapper.stream = curr_streaming
            print("OpenAI Error: " + str(e))
            self.attempt_emergency_save()
            raise e
        finally:
            # make sure the stream is turned off
            self.completion_wrapper.stream = curr_streaming

    def chat(self, user_message: str | dict | Message):
        """Sends the given message to the model and returns the response formatted to return type."""

        self._check_setup()
        self._auto_save_tick()

        user_message = self._process_user_message(user_message)
        self.trim_object.user_message_as_Message = user_message

        try:
            response = self.completion_wrapper.chat(
                self.trim_object.get_finished_chatlog()
            )

        except openai.OpenAIError as e:
            self.logger.critical("OpenAI Error: " + str(e))
            print("OpenAI Error: " + str(e))
            self.attempt_emergency_save()
            raise e
        self.trim_object.assistant_message = response

        return self._format_return(response)

    def reset(self) -> None:
        """Resets the trim chat log object but doesn't reset the completion wrapper object."""
        self.logger.info("Chat Wrapper Reseting")
        self.trim_object.reset()
        self.logger.info("Chat Wrapper Reset")
        if self._check_autosaving():
            self.logger.info("Auto Saving Enabled, resetting auto save counter")
            self._auto_save_counter = 0
            # backup the saves before resetting
            self.rotating_save_handler.backup_saves()
            # deletes all the saves
            self.rotating_save_handler.reset()

    # ================(SETUP/TEMPLATE SYSTEM)================
    """Methods used to set up the ChatWrapper object and manage the template system."""

    def load_template(self, template: dict = None) -> None:
        """Adds a new template to the chat wrapper"""
        self.template = template
        self.logger.info("Template Loaded")

    def auto_setup_from_template(self, template: dict = None) -> None:
        """Creates a new TrimChatLog and ChatCompletionWrapper object from the given template."""
        if self.template is None:
            self.load_template(template)
        if self.template is None:
            raise exceptions.MissingValueError(
                "Chat Wrapper has no template loaded and thus cannot auto setup from template."
            )
        self.model = self.template["model"]

        self.trim_object = TrimChatLog(**self.template["trim_object"])
        self.completion_wrapper = ChatCompletionWrapper(
            API_KEY=self.API_KEY, **self.template["chat_completion_wrapper"]
        )
        if self.stream_handler is not None:
            self.completion_wrapper.add_stream_output_handler(self.stream_handler)

        reminder = self.template["trim_object"].get("reminder", None)
        if reminder is not None:
            self.trim_object.reminder = reminder

        self.logger.info(
            f"Chat Wrapper Auto Setup Complete from Template {self.template['name']} Successfully Completed "
        )

    def reload_completion_from_template(self) -> None:
        """Reloads the completion wrapper from the template. Useful if you want to change the parameters of the completion wrapper, but want to keep the same template."""
        if self.template is None:
            raise exceptions.MissingValueError(
                "Chat Wrapper has no template loaded and thus cannot auto setup from template."
            )
        self.completion_wrapper = ChatCompletionWrapper(
            API_KEY=self.API_KEY, **self.template["chat_completion_wrapper"]
        )
        self.logger.info(
            f"Chat Wrapper Auto Setup Complete from Template {self.template['name']} Successfully Completed "
        )

    def auto_setup(
        self, trim_params: dict = None, completion_params: dict = None
    ) -> None:
        """Sets up the ChatWrapper object with the given parameters.
        Valid parameters:
        Trim Parameters:
            max_tokens: int The maximum number of tokens the model can work with. Typically the max for model, but can be lower to save money(costs are per token).
            max_completion: int The maximum number of tokens the model can use on completion. Used to work out how many tokens can be included in the finished chat log.
            system_prompt: str The system prompt to use for the TrimChatLog object.
            token_padding: int The number of tokens to subtract from the max_tokens to prevent errors in case tokens are counted incorrectly or other parameters take up more tokens than expected.
            max_messages: int The maximum number of messages allowed in the trimmed chat log. Generally this is not reached however if it is the oldest messages are removed. Used to save on resources(trimming off and managing too many messages can be slow). Set to None to disable.
        Completion Parameters:
            temperature: float The temperature to use for completion. Higher temperatures make the model more creative, but also more nonsensical. Must be between 0 and 2(but 0-1 is recommended).
            frequency_penalty: float The frequency penalty to use for
            presence_penalty: float The presence penalty. Sets how much the model should avoid repeating itself on
            completion. Higher values make the model less repetitive, but also more nonsensical. Must be between 0 and 2(but 0-1 is recommended).
            max_tokens: int The maximum number of tokens the model can use on completion.
            top_p: float The top p value to use for completion. Sets the probability of the model choosing the next token. Higher values make the model more creative, but also more nonsensical. Must be between 0 and 1.



        """
        if trim_params is None:
            trim_params = {}
        if completion_params is None:
            completion_params = {}
        self.make_trim_object(**trim_params)
        self.make_chat_completion_wrapper(**completion_params)
        if self.stream_handler is not None:
            self.completion_wrapper.add_stream_output_handler(self.stream_handler)
        self.is_trimmed_setup = True
        self.is_completion_setup = True
        self.logger.info("Chat Wrapper Auto Setup Complete")

    # ================(MAKING/MANAGING OBJECTS)================
    """Used to make and manage the TrimChatLog and ChatCompletionWrapper objects."""

    def make_trim_object(
        self,
        max_tokens: int = 8000,
        max_completion: int = 1000,
        system_prompt: str = None,
        token_padding: int = 500,
        max_messages: int = 400,
    ) -> None:
        """Creates a TrimChatLog object with the given parameters."""
        self.trim_object = TrimChatLog(
            max_tokens=max_tokens,
            max_completion_tokens=max_completion,
            system_prompt=system_prompt,
            token_padding=token_padding,
            max_messages=max_messages,
        )
        # self._sync_models()
        self.message_factory = self.trim_object.get_message_factory()
        self.logger.info("Trim Log object created: " + repr(self.trim_object))

    def make_chat_completion_wrapper(self, **kwargs) -> None:
        """Creates a ChatCompletionWrapper object with the given parameters."""
        self.completion_wrapper = ChatCompletionWrapper(
            model=self.model, API_KEY=self.API_KEY, **kwargs
        )
        self.is_completion_setup = True
        # self._sync_models()
        self.logger.info(
            "Chat Completion Wrapper object created: " + repr(self.completion_wrapper)
        )

    def set_trim_object(self, trim_object: TrimChatLog) -> None:
        """Sets the TrimChatLog object to the given object."""
        if not isinstance(trim_object, TrimChatLog):
            raise exceptions.IncorrectObjectTypeError(
                "Object must be of type TrimChatLog, not " + str(type(trim_object))
            )
        self.trim_object = trim_object
        # self._sync_models()
        self.message_factory = self.trim_object.get_message_factory()
        self.logger.debug("Trim Log object set: " + repr(self.trim_object))
        self.is_trimmed_setup = True

    def set_chat_completion_wrapper(
        self, completion_wrapper: ChatCompletionWrapper
    ) -> None:
        """Sets the ChatCompletionWrapper object to the given object."""
        if not isinstance(completion_wrapper, ChatCompletionWrapper):
            raise exceptions.IncorrectObjectTypeError(
                "Object must be of type ChatCompletionWrapper, not "
                + str(type(completion_wrapper))
            )
        self.completion_wrapper = completion_wrapper

        self.is_completion_setup = True
        self.logger.info("Chat Completion Wrapper object set: ")

        self.logger.debug(
            "Chat Completion Wrapper Info: " + repr(self.completion_wrapper)
        )

    # change TrimChatLog parameters
    def set_trim_token_info(self, **kwargs) -> None:
        """Sets the token info of the TrimChatLog object to the given parameters.
        Valid parameters:
        max_tokens: int The maximum number of tokens the model can work with. Typically the max for model, but can be lower to save money(costs are per token).
        max_messages: int The maximum number of messages allowed in the trimmed chat log. Generally this is not reached however if it is the oldest messages are removed. Used to save on resources(trimming off and managing too many messages can be slow). Set to None to disable.
        max_completion_tokens: Maximum numbers the model can use on completion. Used to work out how many tokens can be included in the finished chat log
        token_padding: int Subtracted from the max_tokens to prevent errors in case tokens are counted incorrectly or other parameters take up more tokens than expected.

        """
        self.trim_object.set_token_info(**kwargs)

    # ==========================================================================
    # ========(SPECIAL VALUES THAT MUST BE THE SAME ACROSS BOTH OBJECTS)========

    @property
    def model(self) -> str:
        """Returns the model of the ChatWrapper object."""
        return self._model

    @model.setter
    def model(self, model: str) -> None:
        """Sets the model of the ChatWrapper object."""
        self._model = model
        self.message_factory = MessageFactory(model=self.model)
        if self.trim_object is not None:
            self.trim_object.model = model
        if self.completion_wrapper is not None:
            self.completion_wrapper.model = model
        # self._sync_models()
        self.logger.info("Model set to " + str(model))

    @property
    def max_tokens(self) -> int:
        """The maximum number of completion tokens for the model. Must be the same across ChatCompletionWrapper and TrimChatLog."""
        self._check_setup("trim")

        if (
            not self.trim_object.max_completion_tokens
            == self.completion_wrapper.parameters.max_tokens
        ):
            self.logger.warning("Max tokens for trim and completion do not match")
            self.trim_object.set_token_info(
                max_completion_tokens=self.completion_wrapper.parameters.max_tokens
            )
        return self.trim_object.max_tokens

    @max_tokens.setter
    def max_tokens(self, max_tokens: int) -> None:
        """Max tokens is changed in both the TrimChatLog and ChatCompletionWrapper objects. Trim calls max tokens max completion tokens."""
        if not isinstance(max_tokens, int):
            raise exceptions.BadTypeError(
                "Max tokens must be of type int, not " + str(type(max_tokens))
            )
        self._check_setup()
        self.trim_object.set_token_info(max_tokens=max_tokens)
        self.completion_wrapper.parameters.max_tokens = max_tokens
        self.logger.debug("Max tokens set to: " + str(max_tokens))

    # =============(CHANGE COMPLETION PARAMETERS)===============================
    def set_chat_completion_params(self, **kwargs) -> None:
        """Sets the parameters of the ChatCompletionWrapper object to the given parameters.
        :max_tokens: int
        :temperature: float
        :presence_penalty: float
        :frequency_penalty: float
        :top_p: float
        "stream: bool
        """
        self._check_setup("completion")
        self.completion_wrapper.set_params(**kwargs)

    @property
    def stream(self) -> bool:
        """Sets whether to stream the output or not."""
        if self.completion_wrapper is None:
            self.logger.warning("Chat Completion Wrapper is not set up")
            return False
        else:
            return self.completion_wrapper.stream

    @stream.setter
    def stream(self, stream: bool) -> None:
        """Sets whether to stream the output or not."""
        if not isinstance(stream, bool):
            raise exceptions.BadTypeError(
                "Stream must be of type bool, not " + str(type(stream))
            )
        self._check_setup("completion")
        self.completion_wrapper.stream = stream
        self.logger.debug("Stream set to: " + str(stream))

    # =====(CONVENIENCE METHODS FOR TRIM CHAT OBJECT )=====
    """Convenience methods for the TrimChatLog object. These methods are just wrappers for the TrimChatLog object's methods.
    Main ones include setters and getters for both the system prompt and reminder.
    As well as setters and getters for the assistant and user messages(If you would like to add a new message to chat log without sending it to the model,you can use these methods)
    
    """

    @property
    def system_prompt(self) -> str:
        """The system prompt of the TrimChatLog object."""
        self._check_setup("trim")

        if self.trim_object.system_prompt is None:
            return None
        return self.trim_object.system_prompt.content

    @system_prompt.setter
    def system_prompt(self, system_prompt: str) -> None:
        """Sets the system prompt to the given system prompt."""
        self._check_setup("trim")
        if not isinstance(system_prompt, str):
            raise exceptions.IncorrectObjectTypeError(
                "System Prompt  must be of type str, not " + str(type(system_prompt))
            )
        self.trim_object.system_prompt = system_prompt
        self.logger.debug("System Prompt set to: " + system_prompt)

    @property
    def reminder(self) -> str:
        """The reminder of the TrimChatLog object."""
        self._check_setup("trim")

        reminder = self.trim_object._reminder_obj.reminder_content
        if reminder is None:
            return None
        return reminder

    @reminder.setter
    def reminder(self, reminder: str) -> None:
        """Sets the reminder to the given reminder.
        This value is treated like the system prompt but it is appended rather than prepended to the chat log.
        """
        self.trim_object.reminder = reminder
        if reminder is None:
            reminder = "None"
        self.logger.debug("Reminder set to: " + reminder)

    @property
    def user_message(self) -> str:
        """The most recent user message as a string."""
        self._check_setup("trim")
        self._auto_save_tick()

        return self.trim_object.user_message

    @user_message.setter
    def user_message(self, user_message: str) -> None:
        """Updates the user message to the given message."""
        self._check_setup("trim")
        self._auto_save_tick()
        if not isinstance(user_message, str):
            raise exceptions.IncorrectObjectTypeError(
                "User Message must be of type str, not " + str(type(user_message))
            )
        self.trim_object.user_message = user_message

    @property
    def assistant_message(self) -> str:
        """Returns the assistant message as a string."""
        self._check_setup("trim")
        return self.trim_object.assistant_message

    @assistant_message.setter
    def assistant_message(self, assistant_message: str) -> None:
        """Sets the assistant message to the given message."""
        if not isinstance(assistant_message, str):
            raise exceptions.IncorrectObjectTypeError(
                "Assistant Message must be of type str, not "
                + str(type(assistant_message))
            )
        self._auto_save_tick()
        self.trim_object.assistant_message = assistant_message

    def get_most_recent_Message(
        self, role: str = "user", pretty: bool = False
    ) -> str | Message:
        """Returns either the most recent user or assistant message as a pretty string  or Message object.
        If Pretty is set to True, the message will be returned as a string stylized using Message's pretty property.
        """
        self._check_setup("trim")
        if role == "user":
            message = self.trim_object.user_message_as_Message
        elif role == "assistant":
            message = self.trim_object.assistant_message_as_Message
        else:
            raise exceptions.BadRoleError(
                "Role must be either 'user' or 'assistant', not " + str(role)
            )

        if pretty:
            return message.pretty
        else:
            return message

    # =================(RETURN TYPE SYSTEM)================
    """Methods for managing the return type of the ChatWrapper object.(What is returned by the chat method)"""
    possible_return_types = {"Message", "str", "string", "dict", "pretty "}

    @property
    def return_type(self) -> str:
        """Gets the return type."""
        return self._return_type

    @return_type.setter
    def return_type(self, return_type: str) -> None:
        """Sets the return type to the given return type."""
        self.logger.info("Return type set to: " + return_type)
        self._return_type = self._check_return_type(return_type)

    # ==================(SAVING/LOADING )======================
    """Methods for persistence. Allows you to save and load the ChatWrapper object. All but the dictionary methods require a save handler to be set up.
    Save handlers use an entry name as a unique identifier and an abstraction for the save content. Just a name that can be used to retrieve the save dictionary.
    
    Be sure to use the check entry name  method before attempting to save or load the ChatWrapper object as the handlers will raise an error if the entry name does not exist.
    """

    def save(self, entry_name: str, overwrite: bool = False) -> None:
        """Saves the ChatWrapper object to the given entry name. If overwrite is set to True, overwrites the entry if it already exists."""
        if not self._check_save_handler():
            raise exceptions.ObjectNotSetupError("Save Handler has not been set up")
        save_dict = self.make_save_dict()
        self.save_handler.write_entry(
            save_dict=save_dict, entry_name=entry_name, overwrite=overwrite
        )
        self.logger.info("Chat Wrapper Saved")

    def load(self, entry_name: str) -> None:
        """Loads the ChatWrapper object from the given entry name."""
        if not self._check_save_handler():
            raise exceptions.ObjectNotSetupError("Save Handler has not been set up")
        save_dict = self.save_handler.read_entry(entry_name)
        self.load_from_save_dict(save_dict)
        self.logger.info("Chat Wrapper Loaded")

    def check_entry_name(self, entry_name: str) -> bool:
        """Returns True if the given entry name exists in the save handler, otherwise returns False."""
        if not self._check_save_handler():
            raise exceptions.ObjectNotSetupError("Save Handler has not been set up")
        return self.save_handler.check_entry(entry_name)

    def delete_entry(self, entry_name):
        """Deletes the entry with the given entry name."""
        self.save_handler.delete_entry(entry_name)

    @property
    def all_entry_names(self) -> list[str]:
        """Returns a list of all the entry names in the save handler."""
        if not self._check_save_handler():
            raise exceptions.ObjectNotSetupError("Save Handler has not been set up")
        return self.save_handler.entry_names
    @property
    def all_non_rotating_entry_names(self) -> list[str]:
        if not self._check_save_handler():
            raise exceptions.ObjectNotSetupError("Save Handler has not been set up")
        saves = self.save_handler.entry_names
        result = []
        for saves in saves:
            if self.rotating_save_handler.is_auto_save(saves):
                continue 
            else:
                result.append(saves)
        return result
    def delete_all_auto_saves(self) -> None:
        if not self._check_save_handler():
            raise exceptions.ObjectNotSetupError("Save Handler has not been set up")
        self.rotating_save_handler.delete_all_saves_and_backups()
            
    def make_save_dict(self):
        """Creates a save dictionary for the ChatWrapper object, that can be used to load the ChatWrapper object."""
        self._check_setup()
        save_dict = {}
        save_dict["trim_object"] = self.trim_object.make_save_dict()
        save_dict["completion_wrapper"] = self.completion_wrapper.make_save_dict()
        save_dict["return_type"] = self.return_type
        save_dict["is_trimmed_setup"] = self.is_trimmed_setup
        save_dict["is_completion_setup"] = self.is_completion_setup
        save_dict["model"] = self.model
        save_dict["timecode"] = str(time.time())
        if self.template is not None:
            save_dict["template"] = self.template
        save_dict["meta"] = {
            "uuid": self.uuid,
            "time_stamp": str(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")),
            "version": self.version,
        }
        self.logger.info("Chat Wrapper Dictionary Created")
        return save_dict

    def load_from_save_dict(self, save_dict: dict) -> None:
        """Loads the ChatWrapper object from the given save dictionary."""
        self.auto_setup()
        self.trim_object.add_chatlog(None)
        # trim will automatically add chat log if its in the save dict. Kinda hacky but we don't want a chat log to be added unless its in the save dict.
        save_dict = self._verify_save_dict(save_dict)
        self.trim_object.load_from_save_dict(save_dict["trim_object"])
        self.completion_wrapper.load_from_save_dict(save_dict["completion_wrapper"])
        self.return_type = save_dict["return_type"]
        self.is_trimmed_setup = save_dict["is_trimmed_setup"]
        self.is_completion_setup = save_dict["is_completion_setup"]
        self.model = save_dict["model"]
        self.uuid = save_dict["meta"]["uuid"]
        self.version = save_dict["meta"]["version"]
        if "template" in save_dict:
            self.template = save_dict["template"]
        self.logger.info("Chat Wrapper Loaded from Save Dictionary")
        self.is_loaded = True

    # ================(HANDLERS)================
    """ Methods for adding handlers to the ChatWrapper object. Handlers are used to save the chat log, stream the chat log, and more. See the handlers module for more info on handlers.
    You can easily create your own handlers by inheriting from the AbstractCWSaveHandler or AbstractStreamOutputHandler classes.
    """

    def add_save_handler(self, save_handler: AbstractCWSaveHandler):
        """Adds the given save handler to the ChatWrapper object. Can be either None(to unset) or of type AbstractCWSaveHandler. (See save_handlers.py for more info on save handlers)"""
        if not isinstance(save_handler, AbstractCWSaveHandler) or save_handler is None:
            raise exceptions.IncorrectObjectTypeError(
                "Save Handler must be of type AbstractCWSaveHandler, not "
                + str(type(save_handler))
            )
        self.save_handler = save_handler
        self.setup_auto_saving()
        self.logger.info("Save Handler Added")

    def add_stream_handler(self, stream_handler: AbstractStreamOutputHandler) -> None:
        """Adds the given stream handler to the ChatWrapper object. Must be either None(to unset) or of type AbstractStreamOutputHandler. (See stream_output_handlers.py for more info on stream handlers"""
        if (
            not isinstance(stream_handler, AbstractStreamOutputHandler)
        ) and stream_handler is not None:
            raise exceptions.IncorrectObjectTypeError(
                "Stream Handler must be of type AbstractStreamOutputHandler, not "
                + str(type(stream_handler))
            )
        if stream_handler is None:
            self.logger.info("Stream Handler is being unset")
        self.stream_handler = stream_handler
        self.logger.info("Stream Handler Added")
        self._add_stream_to_completion_wrapper()

    # ==============(AUTO SAVE FEATURE)================
    """The autosave feature allows for automatic saving every n messages. This is useful for saving the chat log in case of a crash or other error.
    In order to use the autosave feature, you must set up a save handler. This can be done by using the add_save_handler method.
    """

    def set_is_saving(self, is_saving: bool) -> None:
        """Turns autosaving on or off. If autosaving is turned off, the rotating save handler will be unset."""
        if not isinstance(is_saving, bool):
            raise exceptions.BadTypeError(
                "Is Saving must be of type bool, not " + str(type(is_saving))
            )
        self._is_autosaving = is_saving

    def _check_autosaving(self) -> bool:
        """Returns True if autosaving is set up, otherwise returns False."""
        if not self._is_autosaving:
            return False
        if not self._check_save_handler():
            self.logger.warning("Save Handler is not set up, cannot set up autosave")
            return False
        return True

    def setup_auto_saving(self):
        """Syncs the rotating save handler with the save handler and sets up the rotating save handler if it is not already set up."""
        if self.rotating_save_handler is None:
            self.rotating_save_handler = RotatingSave(
                save_handler=self.save_handler,
                entry_name=self._auto_save_entry_name,
                num_entries=self._auto_save_max_saves,
            )
            self.logger.info("Rotating Save Handler Set Up")
        elif isinstance(self.rotating_save_handler, RotatingSave):
            self.rotating_save_handler.add_save_handler(self.save_handler)
            self.rotating_save_handler.set_save_info(
                num_saves=self._auto_save_max_saves,
                save_name=self._auto_save_entry_name,
            )
            self.logger.info("Rotating Save Handler Set Up")
        else:
            self.logger.warning(
                "Rotating Save Handler is not of type RotatingSave, cannot set up autosave"
            )
            return

    def set_auto_save_info(
        self,
        auto_save_frequency: int,
        auto_save_entry_name: str = "auto_save",
        auto_save_max_saves: int = 5,
    ) -> None:
        self._auto_save_frequency = auto_save_frequency
        self.rotating_save_handler.set_save_info(
            num_saves=auto_save_max_saves, save_name=auto_save_entry_name
        )
        # will raise an error if bad type, so we know it is a valid type at this point
        self._auto_save_entry_name = auto_save_entry_name
        self._auto_save_max_saves = auto_save_max_saves

        self.logger.info("Auto Save Info Set")

    def _auto_save_tick(self) -> None:
        """Increments the auto save counter if autosaving is set up and saves if the counter is equal to the auto save frequency. ((A tick is a single chat message))"""
        if not self._check_autosaving():
            return
        self._auto_save_counter += 1
        if self._auto_save_counter >= self._auto_save_frequency:
            self._auto_save_counter = 0
            self.rotating_save_handler.save(self.make_save_dict())
            self.logger.info("Auto Save Sucessful")

    def load_auto_save(self) -> None:
        """Loads the most recent auto save."""
        if not self._check_autosaving():
            return
        most_recent_save = self.rotating_save_handler.find_most_recent_save()
        if most_recent_save is None:
            self.logger.warning("No Auto Save Found")
            return
        if self.check_entry_name(most_recent_save):
            self.load(most_recent_save)
            self.logger.info("Auto Save Loaded")
        else:
            self.logger.error("No auto save found with name " + str(most_recent_save))
            return

    def manual_auto_save(self) -> None:
        """Manually makes an auto save."""
        if not self._check_autosaving():
            raise exceptions.ObjectNotSetupError(
                "In order to manually auto save, auto saving must be set up"
            )
        self.rotating_save_handler.save(self.make_save_dict())

    def auto_setup_autosaving(
        self,
        savehandler: AbstractCWSaveHandler = None,
        entry_name: str = SETTINGS_BAG.AUTO_SAVE_ENTRY_NAME,
        num_entries: int = SETTINGS_BAG.AUTO_SAVE_MAX_SAVES,
        frequency: int = SETTINGS_BAG.AUTO_SAVE_FREQUENCY,
    ) -> None:
        """Automatically sets up autosaving with the given parameters. If no save handler is given, the current save handler will be used. If no save handler is set up, an error will be raised."""
        if savehandler is None:
            if self._check_save_handler():
                savehandler = self.save_handler
            else:
                raise exceptions.MissingValueError(
                    "A save handler must be given if one is not set up"
                )
        else:
            self.add_save_handler(savehandler)
        self.setup_auto_saving()
        self.set_auto_save_info(
            auto_save_frequency=frequency,
            auto_save_entry_name=entry_name,
            auto_save_max_saves=num_entries,
        )
        self._is_autosaving = True
        self.logger.info("Auto Save Set Up")

    # =============================(HELPERS)============================
    """Helper methods for the ChatWrapper object, not intended for public use."""

    def _check_setup(self, setup_type: str = None) -> None:
        """Checks if the given setup type has been set up. Raises an error if it has not been set up."""
        if setup_type is None or setup_type.lower in (
            "trim",
            "chatlog",
            "trimmedchatlog",
        ):
            if not self.is_trimmed_setup:
                if self.trim_object is None:
                    raise exceptions.ObjectNotSetupError(
                        "TrimChatLog object has not been set up."
                    )
                else:
                    self.is_trimmed_setup = True
        if setup_type is None or setup_type.lower in (
            "completion",
            "completionwrapper",
            "chatcompletionwrapper",
        ):
            if not self.is_completion_setup:
                if self.completion_wrapper is None:
                    raise exceptions.ObjectNotSetupError(
                        "ChatCompletionWrapper object has not been set up."
                    )
                else:
                    self.is_completion_setup = True

    # for core methods
    def _process_user_message(self, user_message: str | dict | Message) -> Message:
        """Processes the given user message and returns it as a Message object."""
        if isinstance(user_message, str):
            user_message = self.message_factory(role="user", content=user_message)
            self.logger.debug("User Message created: " + repr(user_message))
        elif isinstance(user_message, dict):
            user_message = self.message_factory(**user_message)
        elif not isinstance(user_message, Message):
            raise exceptions.IncorrectObjectTypeError(
                "Message must be of type str, dict, or Message, not "
                + str(type(user_message))
            )
        return user_message

    def _format_return(self, response: str) -> Union[Message, str, dict]:
        """Formats the given response to the return type."""
        if self.return_type == "Message":
            return self.message_factory(role="assistant", content=response)
        elif self.return_type in ("str", "string"):
            return response
        elif self.return_type == "dict":
            return {"role": "assistant", "content": response}
        elif self.return_type == "pretty":
            msg = self.message_factory(role="assistant", content=response)
            return msg.pretty

    # emergency save system
    def attempt_emergency_save(self) -> str | None:
        """Attempts to make an emergency save of the chat log. Returns the name of the save file if successful, otherwise returns None."""
        print("Attempting to make an emergency save...")
        emergency_save = self._emergency_save()
        if emergency_save is not None:
            print(f"Emergency save successful, saved to {emergency_save}")
            return emergency_save
        else:
            print(
                "Emergency save failed, your chat log has not been saved as there is no save handler to do so."
            )
            print(
                "Attempting to save to emergency_save.json you will either see an error message or the chat log will be saved to emergency_save.json(in the program directory)"
            )
            print(
                "If you see an error message you are extremely unlucky and the chat log has not been saved."
            )
            try:
                with open("emergency_save.json", "w") as f:
                    json.dump(self.trim_object.get_finished_chatlog(), f, indent=4)
                print("Emergency save successful, saved to emergency_save.json")
                return None
            except Exception as e:
                print("That went wrong too....")
                print("We are sort of at the end of line here")

                print(
                    "This is the raw save dict, consult the documentation to see how to load it back in."
                )
                print(
                    "If I forget to add it, you need to use the load_from_save_dict method on the ChatWrapper object to load it back in. I promise its not too complex. "
                )
                print(self.make_save_dict())

    def _emergency_save(self) -> str | None:
        """Makes an emergency save of the chat log, meant to be called when an error occurs."""
        if not self._check_save_handler():
            self.logger.error("Emergency save failed, no save handler")
            return None
        emergency_save_name = f"emergency_save_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.json"
        self.logger.error(f"Emergency save started, saving to {emergency_save_name}")
        self.save(emergency_save_name)
        return emergency_save_name

    # checkers and verifiers
    def _check_return_type(self, return_type: str) -> str:
        """Checks if the given return type is valid. If not raises an InvalidReturnTypeError."""
        if return_type not in self.possible_return_types:
            raise exceptions.InvalidReturnTypeError(
                message=None,
                bad_type=return_type,
                allowed_types=self.possible_return_types,
            )
        return return_type

    def _check_save_handler(self) -> bool:
        """Returns True if a save handler has been set, otherwise returns False."""
        if self.save_handler is None:
            return False
        else:
            return True

    def _has_stream_handler(self) -> bool:
        """Returns True if a stream handler has been set, otherwise returns False."""
        if self.stream_handler is None:
            return False
        return True

    def _add_stream_to_completion_wrapper(self) -> None:
        """Adds the stream handler to the completion wrapper."""
        if self.stream_handler is not None:
            self.completion_wrapper.add_stream_output_handler(self.stream_handler)
            self.logger.info("Stream Handler Added to Completion Wrapper")
        else:
            self.logger.warning(
                "Stream Handler is None, cannot add to completion wrapper"
            )

    def _verify_save_dict(self, save_dict: dict) -> dict:
        """Verifies that the given save_dict is valid. If not raises an BadSaveDictionaryError."""
        required_keys = {
            "trim_object": dict,
            "completion_wrapper": dict,
            "return_type": str,
            "is_trimmed_setup": bool,
            "is_completion_setup": bool,
            "model": str,
            "meta": dict,
        }
        for key in required_keys:
            if key not in save_dict:
                raise exceptions.BadSaveDictionaryError(
                    message="Chat Wrapper save dictionary is missing the key "
                    + str(key)
                )
            if not isinstance(save_dict[key], required_keys[key]):
                raise exceptions.BadSaveDictionaryError(
                    "Chat Wrapper save dictionary key "
                    + str(key)
                    + " must be of type "
                    + str(required_keys[key])
                    + ", not "
                    + str(type(save_dict[key]))
                )
        for key in save_dict["meta"]:
            if not isinstance(save_dict["meta"][key], str):
                raise exceptions.BadSaveDictionaryError(
                    "Chat Wrapper save dictionary meta key "
                    + str(key)
                    + " must be of type str, not "
                    + str(type(save_dict["meta"][key]))
                )

        return save_dict

    # ===============(MISC)================
    def __repr__(self) -> str:
        """Returns the repr of the ChatWrapper object."""
        return (
            "ChatWrapper("
            + "model = "
            + str(self.model)
            + ","
            + "API_KEY = "
            + "exists"
            + ","
            + "return_type = "
            + str(self.return_type)
            + ")"
        )

    def debug(self) -> str:
        """Outputs debug information about the ChatWrapper object."""
        constructor = (
            "ChatWrapper(" + str(self.model) + "API_KEY = " + "exists"
            "," + "return_type = " + str(self.return_type) + ")"
        )
        info = (
            "Chat Wrapper Object with UUID: "
            + str(self.uuid)
            + " and version: "
            + str(self.version)
        )
        msg_list = [constructor, info]
        msg_list.append("Is Trimmed Setup: " + str(self.is_trimmed_setup))
        msg_list.append("Is Completion Setup: " + str(self.is_completion_setup))
        msg_list.append("Is Loaded: " + str(self.is_loaded))
        msg_list.append("Auto Saving: " + str(self._is_autosaving))
        msg_list.append("Auto Save Frequency: " + str(self._auto_save_frequency))
        msg_list.append("Auto Save Counter: " + str(self._auto_save_counter))
        msg_list.append("Trim Object Info:")
        msg_list.append("-" * 20)
        msg_list.append(self.trim_object.__repr__())
        msg_list.append("-" * 20)
        msg_list.append("Completion Wrapper Info:")
        msg_list.append("-" * 20)
        msg_list.append(self.completion_wrapper.__repr__())
        msg_list.append("-" * 20)
        return "\n".join(msg_list)

    def __str__(self):
        return self.trim_object.__str__()


