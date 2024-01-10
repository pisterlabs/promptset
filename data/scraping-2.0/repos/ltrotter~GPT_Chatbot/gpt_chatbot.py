# API call to GPT-4

import os
import openai
import time
import tiktoken
import sys
import colorama
import pyperclip

# set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# set names of files to save conversations to
convo_path = "conversations"
if not os.path.exists(convo_path):
    os.makedirs(convo_path)

# set the colors for the conversation
colorama.init()
mc = 36 # message color
sc = 35 # system color
wc = 33 # warning color

# ~~~~~~~~~~~~~~~~~~~~~~~ CLASSES ~~~~~~~~~~~~~~~~~~~~~~~#
class Conversation:
    """A conversation with the GPT engine.
       A conversation is a list of interactions."""
    
    def __init__(self, system_msg) -> None:
        # set the conversation file based on the time the conversation started
        self.time       = time.localtime()
        self.convo_file = f"{convo_path}/{get_time('file', self.time)}"

        # set default settings
        self.continuing = True
        self.model = "gpt-3.5-turbo" # gpt-4
        self.max_tokens = 1000
        self.temperature = .5

        # set the conversation to an empty list
        self.messages = [{"role": "system", "content": system_msg}]

        # set the token count to 0
        self.token_count = 0

    ## PROPERTY GETTERS
    @property
    def continuing(self):
        """Get whether the conversation is continuing."""
        return self._continuing
    
    @property
    def model(self):
        """Get the model."""
        return self._model
    
    @property
    def max_tokens(self):
        """Get the max tokens ceiling."""
        return self._max_tokens
    
    @property
    def temperature(self):
        """Get the temperature."""
        return self._temperature
        
    ## SETTERS
    @continuing.setter
    def continuing(self, continuing):
        """Set whether the conversation is continuing."""
        #check if continuing can be converted to a boolean
        try:
            # if it can, set it
            continuing = bool(continuing)
            self._continuing = continuing
        except:
            # if it can't, print a warning
            print(colf(f"Warning: continuing must be either True or False (1 or 0). The previous value of {self._continuing} was unchanged", wc))
    
    @model.setter
    def model(self, model):
        """Set the model."""
        if model in ["gpt-4", "gpt-3.5-turbo"]:
            self._model = model
        else:
            print(colf(f"Warning: model must be either 'gpt-4' or 'gpt-3.5-turbo'. The previous value of {self._model} was unchanged", wc))
    
    @max_tokens.setter
    def max_tokens(self, max_tokens):
        """Set the max tokens ceiling."""
        # check if max_tokens can be converted to an integer
        try:
            # if it can, set it
            max_tokens = int(max_tokens)
            self._max_tokens = max_tokens
        except:
            try:
                # if it can't, try to convert it to a float
                max_tokens = float(max_tokens)
                # if it can, round it and print a warning
                self._max_tokens = round(max_tokens)
                print(colf(f"Warning: max_tokens must be an integer. The value was rounded to {self._max_tokens}", wc))
            except:
                print(colf(f"Warning: max_tokens must be an integer. The previous value of {self._max_tokens} was unchanged", wc))
    
    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature."""
        # check if temperature can be converted to a float
        try:
            temperature = float(temperature)
            # if it's between 0 and 1, set it
            if 0 <= temperature <= 1:
                self._temperature = temperature
            # if it's not between 0 and 1, print a warning and set it to either 0 or 1
            else:
                self._temperature = 0 if temperature < 0 else 1
                print(colf(f"Warning: temperature was set to {temperature} but must be between 0 and 1. It was set to {self._temperature}.", wc))
        except:
            print(colf(f"Warning: temperature must be a number. The previous value of {self._temperature} was unchanged", wc))
    
    def get_response(self):
        """Get a response from the GPT engine."""
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = self.messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens - self.token_count
        )

        # get the response text
        response_text = response["choices"][0]["text"]

        # update token count
        self.token_count += response["usage"]["total_tokens"]

        return response
    
    def stream_response(self):
        """Stream a response from the GPT engine."""
        response_text = ""

        for chunk in openai.ChatCompletion.create(
            model = self.model,
            messages = self.messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens - self.token_count,
            stream = True
        ):
            content_response = chunk["choices"][0]["delta"]
            content = content_response.get('content', '')
            if content is not None:
                 response_text += content
                 print(content, flush=True, end='')
        print('\n')

        # update token count
        enc = tiktoken.encoding_for_model(self.model)
        self.token_count += len(enc.encode(response_text))
        self.token_count += len(enc.encode(str(self.messages)))

        return response_text
    
    def handle_command(self, prompt):
        """Handles a command prompted by the user
           valid commands include:
           :temperarure [float] - sets the temperature
           :max_tokens [int] - sets the max tokens ceiling
           :model [model] - sets the model
           :continue - toggles continuing
           :help - shows help
           :quit, :exit, :stop, :end - quits the program"""
        
        # get the command
        command = prompt.split(" ")[0][1:].lower()
        # get the argument
        if len(prompt.split(" ")) > 1:
            argument = prompt.split(" ")[1]
        
        end = False
        new = False
        msg = None

        # handle the command
        if command in ["temperature", "temp", "t"]:
            self.temperature = argument
            msg = f"Temperature set to {self.temperature}.\n"
        elif command == "max_tokens":
            self.max_tokens = argument
            msg = f"Max tokens set to {self.max_tokens}.\n"
        elif command in ["model", "m", "mod"]:
            self.model = argument
            msg = f"Model set to {self.model}.\n"
        elif command == "continue":
            self.continuing = not self.continuing
            msg = f"Continuing set to {self.continuing}.\n"
        elif command in ["q", "quit", "exit", "stop", "end"]:
            msg = f"Conversation ended at {get_time()}\n"
            end = True
        elif command in ["n", "new"]:
            msg = f"Conversation ended at {get_time()}\n"
            new = True
        else:
            self.show_help()
        
        # if the message exists (i.re. everything but help) print the message and write it to message to the conversation file
        if msg:
            print(colf(msg, mc))
            with open(self.convo_file, "a") as f:
                f.write(f"{get_time()}\n{msg}\n")
        # if the command was to end or start a new conversation, do so
        if new:
            main()
        if end:
            sys.exit()  

    def show_help(self):
        """Show help."""
        print(colf("""
        temperature [float] - sets the temperature
        :max_tokens [int] - sets the max tokens ceiling
        :model [model] - sets the model
        :continue - toggles continuing
        :help - shows help
        :quit, :exit, :stop, :end - quits the program
        """, mc))
    
# ~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~#
def get_time(type = "text", t = time.localtime()):
    '''Get the current time in the format "dd/mm/yyyy hh:mm:ss" for text or "yyyymmdd_hhmmss" for file names.'''
    if type == "text":
        return time.strftime("%d/%m/%Y %H:%M:%S", t)
    elif type == "file":
        return time.strftime("%Y%m%d_%H%M%S", t)
    else:
        raise ValueError("type must be either 'text' or 'file'")

def colf(message, color):
    """Formats a message in the message color."""
    return('\033[' + str(color) + 'm' + message + '\033[0m')

def get_prompt(whom = "You", col = mc):
    """Get the next user prompt for the conversation."""
    message = input(colf(whom + ": ", col))
    return check_for_clipboard(message)

def check_for_clipboard(prompt):
    '''Replace {clipboard} or {clip} with the contents of the clipboard'''
    if prompt == "clip":
        prompt = pyperclip.paste()
    else:
        for word in ["{clipboard}", "{clip}"]:
            if word in prompt:
                prompt = prompt.replace(word, pyperclip.paste())
        # Pastes the clipboard in a "code block" (triple backticks)
        if "{cn}" in prompt:
            prompt = prompt.replace("{cn}", f"\n\n'''\n{pyperclip.paste()}\n'''")
    return prompt

# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~#
def main():
    """Main function."""
    # ask for a system message
    system_msg = get_prompt("System message", sc)

    while True:
        # create a new conversation
        C = Conversation(system_msg)
        print(colf(f"Conversation started at {get_time(t = C.time)}", sc))

        # create file to save conversation to
        with open(C.convo_file, "w") as f:
            f.write(f"{get_time(t = C.time)}\nSystem: {system_msg}\n\n")
        
        try:
            # loop until the conversation is over
            while True:
                # Ask for a message and add it to the conversation
                message = get_prompt()
                if message[0] == ":":
                    C.handle_command(message)
                    continue
                
                C.messages.append({"role": "user", "content": message})
                # write the prompt to the conversation file
                with open(C.convo_file, "a") as f:
                    f.write(f"{get_time()}\nYou: {message}\n")

                # stream the response
                print(colf("Bot: ", mc), end='')
                response = C.stream_response()
                # write the response to the conversation file
                with open(C.convo_file, "a") as f:
                    f.write(f"{get_time()}\nBot: {response}\n\n")

                # copy the response to the clipboard
                pyperclip.copy(response)

                # print the max token left
                print(colf(f"{max(C.max_tokens - C.token_count, 0)} tokens left\n", mc))

                if C.max_tokens <= C.token_count or not C.continuing:
                    break
                else:
                    # if the conversation is continuing, add the response to the conversation
                    C.messages.append({"role": "system", "content": response})

        except Exception as e:
            print(e)

            # Print the line the error occurs on 
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        except KeyboardInterrupt:
            sys.exit()

if __name__ == "__main__":
    main()