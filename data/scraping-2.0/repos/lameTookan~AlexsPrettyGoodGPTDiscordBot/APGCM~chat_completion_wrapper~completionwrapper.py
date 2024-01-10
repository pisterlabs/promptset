import chat_completion_wrapper.parameters
from handler.stream_handler import AbstractStreamOutputHandler, StdoutStreamHandler
from log_config import BaseLogger, DEFAULT_LOGGING_LEVEL
import uuid 
import exceptions
import openai 
import time 

class ChatCompletionWrapper:
    version = "2.0.0"
    """
    A simple wrapper for the OpenAI chat completion API.
    Relies on the ModelParameters class to validate parameters and store them.
    Attributes:
        _model: str
        uuid: str
        API_KEY: str(should be set to the OpenAI API key, essential for the chat method)
        parameters: ModelParameters object 
        is_loaded: bool, True if ChatWrapper has been loaded from a save dictionary, False otherwise.
        version: str, version of the ChatCompletionWrapper class.
    Methods:
        Misc:
            get_params: returns a dictionary of the parameters of the model(including those that are None) Uses the ModelParameters class.
            set_params: updates the parameters of the model, takes keyword arguments, uses the ModelParameters class to verify the parameters and store them.
            model get/set: sets the model to use for the chat completion, takes a string.
        Validation:
            _verify_message: verifies that the message dictionary has the correct keys, returns True if it does, False otherwise.
            _verify_messages: verifies that the messages list is valid, returns True if it is, False otherwise.
            _check_save_dict: verifies that the save dictionary is valid, Raises a BadSaveDictionaryError if it is not.
        Core:
            chat: takes a list of message dictionaries, each dictionary should have a 'content' key and a 'role' key, returns a string of the AI response.
        Save/Load:
            make_save_dict: returns a dictionary that can be used to save the state of the ChatCompletionWrapper.
            load_from_save_dict: loads the state of the ChatCompletionWrapper from a save dict, raises a BadSaveDictionaryError if the save dict is invalid.
        Misc:
            __repr__: returns a string representation of the ChatCompletionWrapper object.
            __str__: returns a string representation of the ChatCompletionWrapper object.
    Example Usage:
        wrapper = ChatCompletionWrapper("gpt-4", API_KEY)
        wrapper.set_params(max_tokens = 1000, temperature = 0.9)
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role: "assistant", "content": "I am doing well, how are you?"},
        ]
        response = wrapper.chat(messages)
        print(response)
            
        
    
    """
    def __init__(self, model: str , API_KEY: str, **kwargs ):
        self.logger = BaseLogger(__file__, filename = "ccw.log", identifier="ChatCompletionWrapper", level = DEFAULT_LOGGING_LEVEL)
        self._model = model
        self.uuid = str(uuid.uuid4())
        self.API_KEY = API_KEY
        self.parameters = chat_completion_wrapper.parameters.ModelParameters()
        self.parameters.set_params(**kwargs)
        self.stream = False
        self.stream_handler: AbstractStreamOutputHandler = None
        self.is_loaded = False
        
    @property
    def model(self)-> str:
        """Sets the model to use for the chat completion, takes a string."""
        return self._model
    @model.setter
    def model(self, model: str):
        """Sets the model to use for the chat completion, takes a string."""
        self._model = model
    @property
    def stream(self)-> bool:
        return self.parameters.stream
    @stream.setter
    def stream(self, stream: bool):
        if not isinstance(stream, bool):
            raise exceptions.IncorrectObjectTypeError("stream must be a bool")
        self.parameters.stream = stream
    def get_params(self)-> dict:
        """Returns a dictionary of the parameters of the model(including those that are None)"""
        return self.parameters.get_all_params_dict().update({'model': self.model})
    def set_params(self, **kwargs):
        """Updates the parameters of the model, takes keyword arguments, uses the ModelParameters class to verify the parameters and store them. 
        Valid parameters are:
            max_tokens: int
            temperature: float
            presence_penalty: float
            frequency_penalty: float
            top_p: float
        See ModelParameters class for more information.
        """
        self.parameters.set_params(**kwargs)
    def stream_chat(self, messages: list[dict] ) -> openai.ChatCompletion:
        """Returns a ChatCompletion object directly from the OpenAI API, without any modifications."""
        openai.api_key = self.API_KEY
        tries = 3
        while True:
            try:
                return openai.ChatCompletion.create(model=self.model, messages=self._verify_messages(messages), **self.parameters.get_param_kwargs())

            except openai.OpenAIError as e:
                tries -= 1
                if tries == 0:
                    raise e
                print("Error: {}".format(e))
                print("Retrying {} more times".format(tries))
                time.sleep(5)
                continue
    def chat(self, messages: list[dict]) -> str:
        """Main method for the ChatCompletionWrapper class, takes a list of messages and returns a response as a string. """
        openai.api_key = self.API_KEY
        tries = 3 
        while True:
            try:
                if not self._is_streaming():
                    self.logger.debug(self.parameters.get_param_kwargs())
                    response = openai.ChatCompletion.create(
                        model = self.model,
                        messages = self._verify_messages(messages),
                        **self.parameters.get_param_kwargs()
                    )
                    return response.choices[0].message.content
                else:
                    
                    response = openai.ChatCompletion.create(model=self.model, messages=self._verify_messages(messages),  **self.parameters.get_param_kwargs())
                    response_str = ""
                    for event in response:
                       stop_reason = event.choices[0].finish_reason
                       if stop_reason is None:
                            text = event.choices[0].delta.content
                            response_str += text
                            self.stream_handler.write(text, event)
                       else: 
                            self.stream_handler.done(stop_reason)
                            break
                           
                    return response_str
            except openai.OpenAIError as e:
                tries -= 1
                if tries == 0:
                    raise e
                print("Error: {}".format(e))
                print("Retrying {} more times".format(tries))
                time.sleep(5)
                continue 
                
                
    def _verify_messages(self, messages: list[dict]) -> list[dict]:
        """Verifies that the messages are valid messages, raises a BadMessageError if they are not"""
        if not isinstance(messages, list):
            raise exceptions.BadMessageError("Messages must be a list")
        for message in messages:
            self._verify_message(message)
        return messages
    def _verify_message(self, message: dict) -> dict:
        """Verifies that the message is a valid message, raises a BadMessageError  if it is not"""
        if not isinstance(message, dict):
            raise exceptions.BadMessageError("Message must be a dictionary")
        if "content" not in message:
            raise exceptions.BadMessageError("Message must have a 'content' key")
        if not isinstance(message["content"], str):
            raise exceptions.BadMessageError("Message content must be a string")
        if "role" not in message:
            raise exceptions.BadMessageError("Message must have a 'role' key")
        if not isinstance(message["role"], str):
            raise exceptions.BadMessageError("Message role must be a string")
        return message
    
    def make_save_dict(self) -> dict:
        """Returns a dictionary that can be used to save the state of the ChatCompletionWrapper. """
        return {
            "model": self.model,
            "uuid": self.uuid,
            "parameters": self.parameters.make_save_dict(),
            
        }
    def _check_save_dict(self, save_dict: dict ) -> dict:
        """Checks that the save dict is valid, returns a valid save dict Raises BadSaveDictionaryError if the save dict is invalid."""
        if not isinstance(save_dict, dict):
            raise exceptions.BadSaveDictionaryError("ChatCompletionWrapper save dict must be a dictionary")
        if "model" not in save_dict:
            raise exceptions.BadSaveDictionaryError("ChatCompletionWrapper save dict must have a 'model' key")
        if not isinstance(save_dict["model"], str):
            raise exceptions.BadSaveDictionaryError("ChatCompletionWrapper save dict 'model' key must be a string")
        if not isinstance(save_dict["uuid"], str):
            raise exceptions.BadSaveDictionaryError("ChatCompletionWrapper save dict 'uuid' key must be a string")
        if not isinstance(save_dict["parameters"], dict):
            raise exceptions.BadSaveDictionaryError("ChatCompletionWrapper save dict 'parameters' key must be a dictionary")
        return save_dict
    def load_from_save_dict(self, save_dict: dict) -> None:
        """Loads the state of the ChatCompletionWrapper from a save dict.
        Raises BadSaveDictionaryError if the save dict is invalid.
        """
        save_dict = self._check_save_dict(save_dict)
        self.model = save_dict["model"]
        self.uuid = save_dict["uuid"]
        self.parameters.load_from_save_dict(save_dict["parameters"])
        self.is_loaded = True
    def __repr__(self) -> str:
        msg_list = [
            "ChatCompletionWrapper object with the following attributes:",
            "model: {}".format(self.model),
            "uuid: {}".format(self.uuid),
            "version: {}".format(self.version),
            "is_loaded: {}".format(self.is_loaded),
            "API_KEY:" + "present" if self.API_KEY else "not present",
            "Parameter Object Information:",
            
        ]
        msg_list.extend(self.parameters.__repr__().split("\n"))
        return "\n".join(msg_list)
    #=====(StreamOutputHandler)=====
    def add_stream_output_handler(self, handler: AbstractStreamOutputHandler) -> None:
        """Adds a StreamOutputHandler to the ChatCompletionWrapper"""
        if not isinstance(handler, AbstractStreamOutputHandler) and handler is not None:
            raise exceptions.IncorrectObjectTypeError("StreamOutputHandler must be a subclass of AbstractStreamOutputHandler(or None to unset)")
    
        self.stream_handler = handler
    def _is_streaming(self) -> bool:
        """Returns True if the ChatCompletionWrapper is streaming, False otherwise.
        Streaming means that a stream output handler is set, and the stream parameter is set to True.
        """
        
        if isinstance(self.stream_handler, AbstractStreamOutputHandler) and self.parameters.stream is True:
            
            return True
        else:
            return False
    def __str__(self) -> str:
        return self.__repr__()
    
    
        
    
        
        