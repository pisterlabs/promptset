import time
import json
from typing import List

import tiktoken
import openai

from util import open_file, save_file
from types import Message, Conversation

class GptChat:
    messages: List[Message]
    conversations: List[Conversation]

    def __init__(self, system_prompt_file: str) -> None:
        self.system_prompt = open_file(system_prompt_file)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.system_prompt_tokens = len(self.encoding.encode(self.system_prompt))
        self.messages = []
        self.conversations = []
        self.reset_chat()
        self.total_tokens = 0

    def save_completions(self, file_name):
        if len(self.conversations) == 0:
            return
        text = ''
        for completion in self.conversations:
            text += json.dumps(completion.to_object()) + '\n'
        save_file(file_name, text)

    def add_message(self, message: str, role: str):
        self.messages.append(Message(role, message))

    def reset_chat(self):
        if len(self.messages) > 1:
            self.conversations.append(Conversation(self.messages))
        self.messages = [ ]
        self.add_message(self.system_prompt, "system")

    def get_message_tokens(self) -> int:
        message_tokens = 0
        for m in self.messages:
            message_tokens += len(self.encoding.encode(m.content))
        return message_tokens

    def send(self, message: str, max_tokens: int = 100, config: dict = { }) -> str:
        message_tokens = self.get_message_tokens()
        message_tokens += len(self.encoding.encode(message))
        print(f"Sending chat with {message_tokens} tokens")
        
        if message_tokens >= 4096 - 200:
            raise "Chat Error too many tokens"
        
        self.add_message(message, "user")
        
        defaultConfig = {
            "model": 'gpt-3.5-turbo',
            "max_tokens": max_tokens,
            "messages": Conversation(self.messages).to_object(),
            "temperature": 0.5
        }

        defaultConfig.update(config)
        try:
            res = openai.ChatCompletion.create(**defaultConfig)
        except:
            print('Error when sending chat, retrying in one minute')
            time.sleep(60)
            self.messages = self.messages[:-1]
            self.send(message, max_tokens)
        msg = res.choices[0].message.content.strip()
        print(f"GPT API responded with {res.usage.completion_tokens} tokens")
        self.add_message(msg, "assistant")
        self.total_tokens += res.usage.total_tokens
        return msg
    
class GptCompletion(GptChat):
    def __init__(self, system_prompt: str) -> None:
        super().__init__(system_prompt)

    def complete(self, prompt: str, max_tokens: int = 100, config: dict = {}):
        self.reset_chat()
        self.send(prompt, max_tokens, config)