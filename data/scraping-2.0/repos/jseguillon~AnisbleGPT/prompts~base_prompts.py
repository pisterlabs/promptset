import logging
import openai
import json
import os

class AnsiblePrompt:
    description = "Default chat prompt"
    parameters = [ ]
    command = ""
    model="gpt-3.5-turbo"

    openai_params = {
        'temperature': 0,
        'max_tokens': 1500,
        'request_timeout': 60
    }

    system_message = "You are an Ansible expert. You answer to user discussions."
    conversation_messages = []

    def __init__(self):
        messages_dir = os.environ.get("ANSIBLE_CHAT_MESSAGES_DIR", "./messages/")
        self.messages_file = f"{messages_dir}/{self.command}_messages.json"

    def new_message(self, text, file_list, **user_params):
        logging.debug(f'New message {self.command} - {self.model} with text: {text}, params: {user_params} and files: {file_list}')

        self.add_conversation_message("user", text)
        response = self.send_message()
        self.add_conversation_message("assistant", response)
        self.save_messages()
        return AnsibleChatAnswer(response, file_list)

    def send_message(self):
        messages = [{'role': 'system', 'content': self.system_message}] + self.conversation_messages
        return self.send_openai(messages)

    def send_openai(self, messages):
        try: 
            logging.debug(f'Calling {self.model} with messages: {messages} and params: {self.openai_params}')
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **self.openai_params
            )
            logging.debug(f'GPT-4 response: {response}')
        except Exception as e:
            logging.error(f'GPT-4 error: {e}')
            return "error"
        return response.choices[0]['message']['content'].strip()

    def add_conversation_message(self, role, content):
        self.conversation_messages.append({"role": role, "content": content})

    def get_conversation(self):
        return self.conversation_messages

    def load_messages(self):
        try:
            with open(self.messages_file, "r") as f:
                self.conversation_messages = json.load(f)
        except Exception:
            self.conversation_messages = []

    def save_messages(self):
        with open(self.messages_file, "w") as f:
            json.dump(self.conversation_messages, f, indent=2)

class ChainedPrompt(AnsiblePrompt):
    def __init__(self, sub_prompts):
        super().__init__()
        self.sub_prompts = sub_prompts

    def new_message(self, text, file_list, **user_params):
        self.add_conversation_message("user", text)
        response = text

        for sub_prompt in self.sub_prompts:
            answer = sub_prompt.new_message(response,file_list)
            response = answer.gpt_answer

        self.add_conversation_message("assistant", response)
        return AnsibleChatAnswer(response, file_list)

class OneShotPrompt(AnsiblePrompt):
    def send_message(self):
        messages = [{'role': 'system', 'content': self.system_message}] + [ self.conversation_messages[-1] ]
        return self.send_openai(messages)

class AnsibleChatAnswer:
    def __init__(self, gpt_answer, file_list):
        self.gpt_answer = gpt_answer
        self.file_list = file_list
