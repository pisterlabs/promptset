"""Logic for generating a response from a user's input"""

import openai

class Bobby:
    """The virtual assistant.
    
    Takes in user input and returns a response, either a chat response or a
    function call.
    """

    def __init__(self, system_message, functions, model="gpt-3.5-tubo-0613"):
        """Initializes the virtual assistant.
        
        Args:
            system_message: The message to send to the API to initialize the
                conversation.
            functions: A list of functions to send to the API.
            model: The model to use for the API.
        """
        self.SYSTEM_MESSAGE = system_message
        self.messages = [system_message]
        self.functions = functions
        self.model = model

    def get_response(self, message):
        """Gets a response from the virtual assistant.
        
        Args:
            message: The message to send to the virtual assistant, a chat
                message or a function call. This is in the form of a dict with
                the following keys:
                    role: The role of the message, either "user" or "function".
                    content: The content of the message, either the user's
                        message or the function's response.
        Returns:
            A response from the virtual assistant. This is in the form of a
            dict with the following keys:
                finish_reason: The reason the conversation ended, either
                    "function_call" or "stop".
                message: The message from the virtual assistant, either a chat
                    message or a function call.
        """
        self.messages.append(message)
        if len(self.functions) != 0:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                functions=self.functions
            )['choices'][0]
        else:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
            )['choices'][0]
        return {
            'finish_reason': completion['finish_reason'],
            'message': completion['message']
        }

    def clear_messages(self):
        """Clears the messages from the virtual assistant."""
        self.messages = [self.SYSTEM_MESSAGE]