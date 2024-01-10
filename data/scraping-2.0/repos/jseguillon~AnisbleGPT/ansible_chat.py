import asyncio
import openai
import re
import logging
import json
import os

from prompts.base_prompts import AnsiblePrompt, ChainedPrompt
from prompts.simple_prompts import HelloWorldPrompt, HelloWorld2Prompt
from prompts.chained_prompts import ChainedPrompt1


prompts=[ HelloWorldPrompt, HelloWorld2Prompt, ChainedPrompt1 ]

log_dir = os.environ.get("ANSIBLE_CHAT_LOG_DIR", "./logs")
logging.basicConfig(filename=os.path.join(log_dir, 'debug.log'), level=logging.DEBUG)

class AnsibleChat:
    messages_list = []
    messages_dir = os.environ.get("ANSIBLE_CHAT_MESSAGES_DIR", "./messages")
    work_dir = os.environ.get("ANSIBLE_CHAT_WORK_DIR", "./ansible")

    def __init__(self, prompt_classes):

        for directory in [log_dir, self.messages_dir, self.work_dir]:
            if not os.path.exists(directory): os.makedirs(directory)

        messages_file = os.path.join(self.messages_dir, "messages.json")

        self.prompt_instances = {}

        self.prompt_mapping = {
            prompt_class.command: {
                'class': prompt_class,
                'command': prompt_class.command,
                'description': prompt_class.description,
                'parameters': prompt_class.parameters,
            }
            for prompt_class in prompt_classes
        }

        self.load_files()
        self.load_messages()

    async def chat(self):
        print("Welcome to AnsibleChat powered by... GPT-3 and GPT-4! Type '/exit' to end the chat.")
        print("Known commands:")
        for command, data in self.prompt_mapping.items():
            params = ', '.join(data['parameters']) if data['parameters'] else 'None'
            print(f"- {command}: {data['description']} (Parameters: {params})")
        print("")

        while True:
            user_input = ""
            try:
                user_input = input("You: ")
                while True:
                    line = input()
                    if not line:
                        break
                    user_input += '\n' + line
            except KeyboardInterrupt:
                user_input == ""
                print("")

            if user_input != "":
                if user_input.lower().startswith('/exit'):
                    exit(0)

                command_name, user_params, text = self.parse_input(user_input)
                
                response = await self.handle_prompt(command_name, user_params, text)
                print("AnsibleChat:", response)

                # Save prompt messages to their respective files when exiting the chat
                for prompt_instance in self.prompt_instances.values():
                    prompt_instance.save_messages()

    def parse_input(self, user_input):
        command_name = None
        user_params = {}
        text = ""

        parts = user_input.split(maxsplit=1)
        if parts and parts[0].startswith('/'):
            command_name = parts[0]
            if len(parts) > 1:
                remaining_text = parts[1]

                # Match parameters with the pattern: key:'value'
                pattern = re.compile(r"(\w+):'([^']+)'")
                matches = pattern.finditer(remaining_text)

                for match in matches:
                    key, value = match.groups()
                    user_params[key] = value

                # Remove matched parameters and leading/trailing whitespaces from the text
                text = pattern.sub('', remaining_text).strip()
        else:
            text = user_input

        return command_name, user_params, text

    async def handle_prompt(self, command_name, user_params, text):
        if command_name is None:
            prompt_class = AnsiblePrompt
            command_name = 'ansible_prompt'  # Add a name for the default AnsiblePrompt
        else:
            prompt_data = self.prompt_mapping.get(command_name, {'class': AnsiblePrompt})
            prompt_class = prompt_data['class']

        # Use an existing instance or create a new one
        if command_name not in self.prompt_instances:
            if command_name not in self.prompt_instances:
                self.prompt_instances[command_name] = prompt_class()

        prompt_instance = self.prompt_instances[command_name]
        answer = prompt_instance.new_message(text, self.file_list, **user_params)

        self.file_list = answer.file_list
        self.save_files()

        return answer.gpt_answer

    def add_conversation_messages(self, messages):
        self.messages_list.extend(messages)
        self.save_messages()

    def load_messages(self):
        try:
            with open(self.messages_file, "r") as f:
                self.messages_list = json.load(f)
        except Exception:
            self.messages_list = []

    def save_messages(self):
        with open(self.messages_file, "w") as f:
            json.dump(self.messages_list, f, indent=2)

    def save_files(self):
        for file_info in self.file_list:
            file_path = os.path.join(self.work_dir, file_info['path'])
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "w") as f:
                f.write(file_info['content'])

    def load_files(self):
        self.file_list = []

        # Define a helper function to recursively load files
        def _load_files_from_dir(directory):
            for entry in os.scandir(directory):
                if entry.is_file():
                    file_path = os.path.join(directory, entry.name)
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Remove the WORK_DIR prefix from the file path
                    relative_path = os.path.relpath(file_path, self.work_dir)
                    self.file_list.append({'summary': '', 'path': relative_path, 'content': content})
                elif entry.is_dir():
                    _load_files_from_dir(entry.path)

        _load_files_from_dir(self.work_dir)

if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]

    ansible_chat = AnsibleChat( prompts )
    asyncio.run(ansible_chat.chat())
