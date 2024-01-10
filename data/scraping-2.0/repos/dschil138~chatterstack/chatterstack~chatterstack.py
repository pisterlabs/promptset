
import openai

class Chatterstack:
    def __init__(self, user_defaults=None, existing_list=None):
        """Initialize the Chatist class with optional user default values & existing list of dictionaries, if any."""
        self.config = {
            key: value
            for key, value in (user_defaults or {}).items()
            if key in {"MODEL", "TEMPERATURE", "TOP_P", "FREQUENCY_PENALTY", "PRESENCE_PENALTY", "MAX_TOKENS", "STOP", "STREAM", "LOGIT_BIAS"}
        }
        if existing_list is None:
            self.list = []
        else:
            self.list = existing_list

        self.debug = False
        self.max_length = 4
        self.system_index = -1
        self.system_lock_index = None

        self.last_call_prompt_tokens=0
        self.last_call_full_context_prompt_tokens=0
        self.last_call_completion_tokens=0
        self.last_call_tokens_all=0

        self.prompt_tokens_total=0
        self.assistant_tokens_total=0
        self.tokens_total_all=0

    def __str__(self):
        """Return a string representation of the conversation."""
        return str(self.list)

    def __len__(self):
        """Return the number of messages in the conversation."""
        return len(self.list)

    def __getitem__(self, index):
        """Return the message at the specified index."""
        if index < 0 or index >= len(self.list):
            raise IndexError("Index out of range")
        return self.list[index]
    
    def add(self, role, content):
        new_dict = {"role": role, "content": content}
        self.list.insert(len(self.list), new_dict)
    
    def add_system(self, content):
        """Add a system message with specified content to the end of the conversation."""
        self.add("system", content)

    def add_assistant(self, content):
        """Add an assistant message with specified content to the end of the conversation."""
        self.add("assistant", content)
        
    def add_user(self, content):
        """Add a user message with specified content to the end of the conversation."""
        self.add("user", content)

    def user_input(self, prefix="USER: "):
        """Prompt the user for input and add it to the end of the conversation."""
        user_input = input(prefix)
        self.add_user(user_input)
        return user_input

    def move_system_to(self, index):
        system_index = -1
        system_count = 0
        for i, d in enumerate(self.list):
            if d["role"] == "system":
                system_count += 1
                if system_count > 1:
                    print("More than one 'system' dict found")
                    return
                system_index = i
        if system_index == -1:
            print("No 'system' dict found")
            return
        if index < 0 or index >= len(self.list):
            print("Index out of range")
            return
        system_dict = self.list.pop(system_index)
        self.list.insert(index, system_dict)
        self.update_system_index()
        # Ensure the "system" dictionary stays as close as possible to the system_lock_index
        if self.system_lock_index is not None:
            if self.system_index > self.system_lock_index:
                self.move_system_to(self.system_lock_index)
            elif self.system_index < self.system_lock_index:
                self.move_system_to(min(self.system_lock_index, len(self.list) - 1))

    def move_system_to_end(self, minus=0):
        if minus < 0:
            print("Minus value cannot be negative")
            return
        target_index = len(self.list) - minus
        self.move_system_to(target_index)
        self.update_system_index()

    def set_max_length(self, max_length):
        self.max_length = max_length

    def trim_to_max_length(self):
        if self.max_length is not None:
            if self.max_length <= 1:
                self.list = [self.list[self.system_index]]
                self.system_index = 0
                return
            while len(self.list) > self.max_length:
                if self.system_index > 0:
                    self.remove_from_start(1)
                elif len(self.list) > 1:
                    self.remove_from_start(2)
                else:
                    break
                self.update_system_index()
                # Get system message as close as possible to its original position
                if self.system_lock_index is not None:
                    if self.system_index > self.system_lock_index:
                        self.move_system_to(self.system_lock_index)
                    elif self.system_index < self.system_lock_index:
                        self.move_system_to(min(self.system_lock_index, len(self.list) - 1))

    def update_system_index(self):
        for i, d in enumerate(self.list):
            if d["role"] == "system":
                self.system_index = i
                return
        self.system_index = -1

    def set_system_lock_index(self, index):
        if index < 0:
            index = len(self.list) + index - 1
        
        if index < 0 or index >= len(self.list):
            print("Index out of range")
            return
        
        self.system_lock_index = index
        self.move_system_to(index)



    def send_to_bot(self, **kwargs):
        """Send the conversation to the OpenAI API and append the response to the end of the conversation. Uses 3.5-turbo by default."""
        self.trim_to_max_length()
        model = kwargs.get("model", self.config.get("model", "gpt-3.5-turbo"))
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.8))
        top_p = kwargs.get("top_p", self.config.get("top_p", 1))
        frequency_penalty = kwargs.get("frequency_penalty", self.config.get("frequency_penalty", 0))
        presence_penalty = kwargs.get("presence_penalty", self.config.get("presence_penalty", 0))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 200))
        stop = kwargs.get("stop", self.config.get("stop", None))
        stream = kwargs.get("stream", self.config.get("stream", False))
        logit_bias = kwargs.get("logit_bias", self.config.get("logit_bias", {}))

        response = openai.ChatCompletion.create(
            model=model,
            messages=self.list,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream,
            logit_bias=logit_bias,
        )
        self.add_assistant(response.choices[0].message.content.strip())

        api_usage = response['usage']
        self.last_call_full_context_prompt_tokens = int((api_usage['prompt_tokens']))
        self.last_call_completion_tokens = int((api_usage['completion_tokens']))
        self.last_call_tokens_all = int((api_usage['total_tokens']))

        self.last_call_prompt_tokens = self.last_call_full_context_prompt_tokens - self.tokens_total_all

        self.prompt_tokens_total += self.last_call_full_context_prompt_tokens
        self.assistant_tokens_total += self.last_call_completion_tokens
        self.tokens_total_all += self.last_call_tokens_all
        return self


    def remove_from_end(self, count):
        """Remove N messages (count) from the end of the list."""
        if count < 0:
            print("Count must be a non-negative integer")
            return
        self.list = self.list[:-count] if count < len(self.list) else []


    def remove_from_start(self, count):
        """Remove N messages (count) from the start of the list."""
        if count < 0:
            print("Count must be a non-negative integer")
            return
        self.list = self.list[count:] if count < len(self.list) else []


    def insert(self, index, role, content):
        """Insert a message at the specified index with the given role and content."""
        if role not in ["system", "assistant", "user"]:
            print("Invalid role")
            return
        if index < 0 or index > len(self.list):
            print("Index out of range")
            return
        self.list.insert(index, {"role": role, "content": content})


    def move_system_message(self, index, from_end=False):
        """Move the system message to the specified index. Only works if conversation contains one system message."""
        system_index = -1
        system_count = 0
        for i, d in enumerate(self.list):
            if d["role"] == "system":
                system_count += 1
                if system_count > 1:
                    print("More than one 'system' dict found")
                    return
                system_index = i
        if system_index == -1:
            print("No 'system' dict found")
            return
        if from_end:
            index = len(self.list) - 1 - index
        if index < 0 or index >= len(self.list):
            print("Index out of range")
            return
        system_dict = self.list.pop(system_index)
        self.list.insert(index, system_dict)


    def find_message_containing(self, substring, check_lock=True):
        matching_indices = [i for i, d in enumerate(self.list) if substring in d["content"]]
        if len(matching_indices) == 0:
            print("Warning: No message containing the substring was found")
            return -1, False
        elif len(matching_indices) > 1:
            print("Warning: Multiple messages containing the substring were found")
            return -1, False
        else:
            message_index = matching_indices[0]
            is_locked = message_index == self.system_lock_index
            if is_locked and check_lock:
                print("Warning: The message containing the substring is locked")
            return message_index, is_locked

    def is_locked_by_substring(self, substring):
        _, is_locked = self.find_message_containing(substring, check_lock=False)
        return is_locked

    def move_message_containing(self, substring, index):
        if index < 0 or index >= len(self.list):
            print("Index out of range")
            return
        message_index, is_locked = self.find_message_containing(substring)
        if message_index == -1:
            return
        if is_locked:
            return
        message_dict = self.list.pop(message_index)
        self.list.insert(index, message_dict)

    def remove_message_containing(self, substring):
        message_index, is_locked = self.find_message_containing(substring)
        if message_index == -1:
            return
        if is_locked:
            return
        del self.list[message_index]
        self.dbprint(f'1 message containing the "{substring}" was removed')
    
    
    


    @property
    def last_message(self):
            return self.list[-1]["content"]
    
    @property
    def last_system_message(self):
        for message in reversed(self.list):
            if message["role"] == "system":
                return message["content"]
        return None  
    
    @property
    def last_user_message(self):
        for message in reversed(self.list):
            if message["role"] == "user":
                return message["content"]
        return None  
    
    @property
    def last_assistant_message(self):
        for message in reversed(self.list):
            if message["role"] == "assistant":
                return message["content"]
        return None  


    def print_last_message(self, prefix="ASSISTANT: ", lines_before=1, lines_after=1):
        """Print the last message in the conversation."""
        for i in range(lines_before):
            print()
        content = self.list[-1]["content"]
        print(f"{prefix}{content}")
        for i in range(lines_after):
            print()

    def print_total_tokens(self):
        """Print the total number of tokens used in the conversation."""
        print(f"Total tokens: {self.tokens_total_all}")

    def print_formatted_conversation(self):
        for d in self.list:
            print(f'{d["role"].capitalize()}: {d["content"]}')


    def summary(self):
        """Return a summary of the conversation."""
        summary_dict = {
            "total_messages": len(self.list),
            "prompt_tokens": self.prompt_tokens_total,
            "assistant_tokens": self.assistant_tokens_total,
            "total_tokens": self.tokens_total_all,
        }
        return summary_dict
    
    def dbprint(self, message):
        if self.debug:
            print(message)

    def clear(self):
        """Clear the conversation."""
        self.list = []






