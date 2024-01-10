import json, tiktoken, os 
from openai import OpenAI, APIConnectionError, AuthenticationError
from configuration import ConfigManager

class ConversationLogic:
    def __init__(self, config_manager):
        """Initializes the ConversationLogic object.

        This constructor sets up the configuration settings, the user's API key, and other requirements/tools for communication with OpenAI's GPT API. """

        self.config_manager = config_manager
        self.config=self.config_manager.config # set default config 

        # starts an instance of OpenAI's API client using the api key
        self.api_key=self.config.get('OPENAI_API_KEY', 'YOUR_DEFAULT_API_KEY_HERE')
        self.client = OpenAI(api_key=self.api_key) 

        # sets the filename given from the configs. This will initially be conversation.json in data/
        self.filename=self.config.get('filename', os.path.join('data', 'conversation.json')) 
        self.model = self.config.get('model', 'gpt-3.5-turbo-1106') 
        self.system_message = self.config.get('system_message', 'You are an assistant providing help with any issues.') 
        self.user_message = self.config.get('user_message','What can you help me with today?') 
        self.assistant_message = self.config.get('assistant_message', 'Hi, how can I help you today?')
        self.max_tokens = self.config.get('max_tokens', 750)

    def chat_gpt(self, user_input):
        """Performs the API call, and inputs the given user input from the GUI to perform the call.
        Args:
            user_input (str): The user's input for the conversation, from the gui input.

        Returns:
            Tuple[str, None or str]: A tuple containing the GPT response and any potential error message. The second element is None if there are no errors.
        
        'messages' is held inside a dictionary. its key is 'messages' and its value is a list '[]' of dictionaries.
        Each dictionary in the list represents a message in the entire conversation
        Each message in the list is a dictionary with two key-value pairs, "role" and "content"
            Role: specifies the role of the entity sending the message. This can either be 'system', 'user', or 'assistant' 
            Content: specifies the context of the message given by the entity (role)

        Remember that the GPT reads the length of 'messages' when trying to providing responses. Basic conversation trimming is performed to reduce the size of input calls in lengthy conversations.       
        Below is an example message '[]' held in the .json 
            {
                "messages": [
                    {
                    "role": "system",
                    "content": "You are an assistant providing help for any task, utilizing context for the best responses"
                    },
                    {
                    "role": "user",
                    "content": "What can you help me with today?"
                    },
                    {
                    "role": "assistant",
                    "content": "Hi there! How can I help you today?"
                    }
                ]
            }  
        """

        messages = self.load_conversation().get('messages', []) 
        new_input_tokens = self.count_tokens_in_messages([{"role": "user", "content": user_input}], model=self.model) # calculates the ~amount of input tokens prior to the API call
        remaining_tokens = self.max_tokens - new_input_tokens # This is a prompt safeguard that handles (all) large user inputs. If the user's prompt is large, the conversation is truncated more harshly to fit within the token limit. This helps reduce costs slightly, at the cost of reducing prior context for the GPT. 
        print(f"\n~ input tokens: {new_input_tokens} ~ remaining tokens: {remaining_tokens}")

        messages = self.trim_conversation_history(messages, remaining_tokens) # Performs the conversation truncation, sends in conversation and the tokens left to use. This new message holds what the api call can handle, and omits the oldest message according to the tokens allowed
        messages.append({"role": "user", "content": user_input }) # appends the newest message to the conversation
        # IMPORTANT: Due to the trim function, chatGPT may lose context of the system message and early context. In the future, introduce better truncation methods (such as summation) 

        try:
            response = self.client.chat.completions.create( # This is the client API call to OpenAI
                model=self.model, # inputs current model type 
                messages=messages, # inputs the given conversation (truncated)
                max_tokens=self.max_tokens, # a "limiter" that helps truncate conversations
            )

            # These are return statements from the API (look at documentation for more info). These are helpful for future logging and debugging. 
            self.total_tokens_used = response.usage.total_tokens
            self.input_tokens = response.usage.prompt_tokens
            self.response_tokens = response.usage.completion_tokens
            self.model_type = response.model
            self.stop_reason = response.choices[0].finish_reason
            print(f"Total tokens used for API call: {self.total_tokens_used} | Total Input: {self.input_tokens} | Total Response: {self.response_tokens}\nModel Used: {self.model_type} | API Stop Reason: {self.stop_reason}")

            response = response.choices[0].message.content # this is the API call to get the latest gpt response 
            self.update_conversation(user_input, response) # updates the conversation with the latest input and response

            return response, None # response is returned to display in gui, None is returned to signal no errors. 
        except AuthenticationError as auth_error:
            return None, (str(auth_error))
        except APIConnectionError as conn_error:
            return None, (str(conn_error))

    def load_conversation(self, filename=None): 
        """Attempts to load the conversation from a given .json file 
        Args:
            filename (str, optional): The path to the conversation JSON file. Defaults to None.

        Returns:
            dict: The loaded conversation data held as a dictionary. """

        if filename is None:
            filename = self.filename # if there is no file found, it is given the configured path. Its default value is data\conversation.json 

        try:
            with open(filename, 'r') as file: 
                loaded_conversation = json.load(file) 
                if filename != self.filename: # if the given file path (conversation) does not match the current file path, a new file path is set to the filepath.
                    self.filename = filename  # Update the filepath if a different file is loaded
                return loaded_conversation # returns the given conversation of the file 
        except FileNotFoundError:  
            print(f"Conversation file not found. Creating a new one.")
            # implement logic for fixing errors in finding the default conversation.json file. 

    def update_conversation(self, user_input, gpt_response):
        """Update the conversation state with the latest user input and GPT response.

        Args:
            user_input (str): The user's input.
            gpt_response (str): The GPT response.
        """
        messages = self.load_conversation().get('messages', [])

        # Update with new information
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": gpt_response})

        # Save the updated state
        self.save_conversation_to_file(self.filename, messages)

    def save_conversation_to_file(self, filename, messages): 
        """Save the conversation to a JSON file.
        Args:
            filename (str): The path to save the conversation JSON file.
            messages (list): The list of messages to be saved. """

        with open(filename, 'w') as file:
            json.dump({"messages": messages}, file)

    def reset_conversation(self):
        """Reset the conversation to a default prompt state.

        This includes setting up a default system message, user message, and assistant message."""
        messages = [        
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message},
            {"role": "assistant", "content": self.assistant_message}
        ]
        # Save the updated state to the current file.
        self.save_conversation_to_file(self.filename, messages)

    def count_tokens_in_messages(self, messages, model):
        """Count the number of tokens in a list of messages. This method is provided by tiktoken (import)
        Args:
            messages (list): List of messages in the conversation.
            model (str): The GPT model being used.

        Returns:
            int: The total number of tokens in the given messages. """
        
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == model: # change model type here if testing a specific model 
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # If there's a name, the role is omitted
                        num_tokens += -1  # Role is always required and always 1 token
            num_tokens += 2  # Every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""count_tokens_in_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        
    def trim_conversation_history(self, messages, remaining_tokens):
        """Trims the last message to fit within the maximum token limit if token limit is hit.

        Args:
            messages (list): List of messages in the conversation.
            remaining_tokens (int): The maximum number of tokens allowed within the conversation (a total of input + output).

        Returns:
            list: The trimmed list of messages that fits within the token limit. """
        tokens_used = 0 # initial token amount
        truncated_messages = [] # new list to hold conversation 
 
        for message in reversed(messages): # REVERSES order of reading messages, looking at the newest information in the conversation first. (appends newest -> oldest in the conversation)
            message_tokens = self.count_tokens_in_messages([message], model=self.model) # calculates the current ~amount of tokens in the current message, using the model type (different amount of token costs)
            if tokens_used + message_tokens <= remaining_tokens: # if the tokens used (tokens in the conversation that accumulate during the loop) + the amount of the current message is less than the max_tokens allowed by the api call, that message is added to this conversation
                truncated_messages.insert(0, message) #inserts the newest message to the new conversation list 
                tokens_used += message_tokens # increases the tokens used (in the conversation) per message added to the list (Reversed)
            else:
                break # stops adding messages if the next message were to exceed the token limit provided

        return truncated_messages
    
    def update_configs(self, new_settings):
        """Abstract class for updating the configs through the config manager
        
        Config file is changed within configuration.py. 
        Add new or updated self. variabvles here to implement the config changes.
        """
        self.config_manager.update_configs(new_settings)
        self.client = OpenAI(api_key=self.config.get('OPENAI_API_KEY')) # client is initiated with new API key. 
        self.model = self.config.get('model') 
        self.system_message = self.config.get('system_message') 
        self.max_tokens = self.config.get('max_tokens')
        # Update any other logic in ConversationLogic as needed
        
    
