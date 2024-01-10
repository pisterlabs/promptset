# Raphael Fortuna (raf269) 
# Rabail Makhdoom (rm857) 
# Final Project Report
# Lab 403, Lab Section:  4:30pm-7:30pm on Thursdays 

import os
import openai
from importlib.machinery import SourceFileLoader

prePath = "/home/pi/"

# use direct path to avoid importing issues when running code from different directories
AICorePaths = prePath + "RoboGPT/src/AI_src/"

# import libraries
try:
    utils = SourceFileLoader("utils", AICorePaths + "utils.py").load_module()
except:
    # if the file is not found, try to import from the current directory - when not running on the pi
    import utils

# get the key
openai.api_key = os.getenv("ROVER_OPENAI_KEY")

class token_history:
        """ Class to keep track of the token usage for each request """
    
        def __init__(self, prompt_tokens = 0, completion_tokens = 0, total_tokens = 0):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens
    
        def __str__(self):
            return "Prompt tokens: " + str(self.prompt_tokens) + ", Completion tokens: " + str(self.completion_tokens) + ", Total tokens: " + str(self.total_tokens)

class openai_chat:

    def __init__(self, system_prompt = "You are a helpful assistant.", max_tokens = 100, model_name = "gpt-3.5-turbo", debug = False):
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.debug = debug

        # if the chat has started
        self.chat_started = False

        # three roles for the messages, user, assistant, and system
        # system is the initial prompt or new information just to the system
        # user is the user's message
        # assistant is the assistant's (ChatGPT's) response
        self.base_message_prompt = {"role": "system", "content": self.system_prompt}

        # total tokens used by all the requests
        self.tokens_used = 0

        # keep track of the token usage for each request
        self.token_histories = []

        # back and forth messages between the user and the assistant
        self.messages = []

        # add the initial system prompt to setup the conversation
        self.messages.append(self.base_message_prompt)
    
    def _reset_chat(self):
        """Reset the chat to the initial state"""

        self.chat_started = False
        self.messages = []
        self.tokens_used = 0
        self.token_histories = []
        self.messages.append(self.base_message_prompt)

    def _add_user_message(self, message: str):
        """Add a user message to the conversation"""

        self.messages.append({"role": "user", "content": message})

    def _system_chat(self, system_text):
        """ add a system message to the text """
            
        self.messages.append({"role": "system", "content": system_text})

    def _add_assistant_message(self, message: str):
        """ Add an assistant message to the conversation
        
        example format direct from completion.choices[0].message
        {
        "content": "A bright, glowing orb,\nGuiding tides and lovers' hearts,\nMoon's magic above.",
        "role": "assistant"
        }
        """
        self.messages.append({"role": "assistant", "content": message})

    def _process_message_from_assistant(self, message: dict):
        """Process a message recieved from the assistant"""

        message_content = message["content"]

        # add the message to the conversation
        self._add_assistant_message(message_content)

        # process the message and return a response
        return message_content
    
    def _update_tokens_used(self, usage):
        """Update the total tokens used by the conversation"""

        self.tokens_used += usage.total_tokens

        self.token_histories.append(token_history(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens))

        if self.debug:
            with utils.ColorText("yellow") as colorPrinter:
                colorPrinter.print(str(usage))
                colorPrinter.print("Total tokens used: " + str(self.tokens_used))
                colorPrinter.print("Prompt tokens used: " + str(usage.prompt_tokens))
                colorPrinter.print("Completion tokens used: " + str(usage.completion_tokens))

    def get_response(self, message: str, system = False):
        """Get a response from the assistant to the user message

        return the response from the assistant

        """

        if system:
            self._system_chat(message)

        else:
            # add the user message to the conversation
            self._add_user_message(message)

        # create the completion request
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            max_tokens = self.max_tokens,
            # default values, unmodified
            # temperature = 1,
            # top_p = 1,
            # frequency_penalty = 0,
            # presence_penalty = 0,
            # stream = False,
            # logit_bias = {},
            # n = 1,
            messages = self.messages
        )

        # update the total tokens used by the conversation
        self._update_tokens_used(completion.usage)

        # get the first response
        response = completion.choices[0].message

        # process the response
        return self._process_message_from_assistant(response)
    
    def init_chat(self):
        """Start a chat with the assistant and display the system prompt"""

        if not self.chat_started:
            print("System prompt: " + self.system_prompt)
            self.chat_started = True

    def looping_test_chat(self):
        """Start a looping chat with the assistant"""

        self.init_chat()

        while True:
            # get the user message
            message = input("User: ")

            # get the response
            response = self.get_response(message)

            utils.color_printer("green", "Assistant: " + response)
            print()
            print()

    def voice_chat(self, input_text, extra = True, system = False):
        """ function to facilitate voice chat """

        self.init_chat()

        # get the response
        response = self.get_response(input_text, system=system)

        # if should print the response
        if extra:
            if not system:
                with utils.ColorText("blue") as colorPrinter:
                    # print the user message
                    colorPrinter.print("User: " + input_text)

            else:
                with utils.ColorText("red") as colorPrinter:
                    # print the user message
                    colorPrinter.print("System: " + input_text)

            with utils.ColorText("green") as colorPrinter:
                # print the response
                colorPrinter.print("Assistant: " + response)

        return response


if __name__ == "__main__":
    chat_instance = openai_chat()
    chat_instance.looping_test_chat()
