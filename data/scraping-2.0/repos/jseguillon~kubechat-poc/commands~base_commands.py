# Default command plus OneShotCommand, ChainedCommand and also KubeChatAnswer (with sanitize mecanism) interfaces
import logging
import openai
import json
import os
import re
import sseclient
import requests

class KubeCommand:
    description = "Default chat prompt"
    parameters = [ ]
    command = ""
    model="gpt-3.5-turbo"
    stream = False
    sanitize_code_answer = True
    include_items = False
    openai_params = {
        'temperature': 0,
        'max_tokens': 3000,
        'request_timeout': 60,
        'stream': stream
    }

    system_message = """Convert the challenge prompt into a list of actions grouped by intent.

Example: "Challenge: explain how to deploy <app> then do it for me. then add 2Gi disk", replies:
[ {'explain': {'user': 'how to to deploy <app> ?', 'assistant_followup': 'do you want me to create Deployment and Service for <app> for you ?'}},
{'create': {'user': 'create a new Deployement named <app>'}},
{'create': {'user': 'create a new Service for Deployment named <app>'}},
{'create': {'user': 'create a new PersistentVolumeClaim 2Gi for Deployment named <app>'}} ]

Dont ouput more details, nor item spec, only intent and user challenge.

Choose intent from list: ['create','update', 'explain','delete','command','logs','troubleshoot','kill','chat','apply','status']
Choose 'explain' only if user explicitly ask for help. Keep original user for 'explain' nor 'chat': set user to original user challenge.

Now you only reply python array, one change at a time:
"""

    def __init__(self):
        self.conversation_messages = []
        messages_dir = os.environ.get("ANSIBLE_CHAT_MESSAGES_DIR", "./messages/")
        self.messages_file = f"{messages_dir}/{self.command}_messages.json"

    def new_message(self, text, items, **user_params):
        logging.debug(f'New message {self.command} - {self.model} with text: {text}, params: {user_params} and files: {items}')

        if self.include_items:
            target_items = self.flatten_array(items.copy())
            new_message=f"# Python\n# current_items={target_items}\n# action: {text} +\n # Now reply no comment, nor note, nor explanation but only pure Python\n:"
        else: new_message=text

        self.add_conversation_message("user", new_message)
        response = self.send_message()
        answer = KubeChatAnswer(response, items)

        if self.sanitize_code_answer: answer.sanitize_code_answer()

        self.add_conversation_message("assistant", answer.gpt_answer)
        return answer

    def send_message(self):
        messages = [{'role': 'system', 'content': self.system_message}] + self.conversation_messages
        if self.stream: 
            return self.send_openai_stream(messages) 

        answer = self.send_openai_no_stream(messages)
        return answer

    def send_openai_stream(self, messages):
        try: 
            response =  openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **self.openai_params
            )
            return self.print_openai_stream(response)
        except Exception as e:
            logging.error(f'error: {e}')
            return ""
        
    def send_openai_no_stream(self, messages):
        try: 
            logging.debug(f'Calling {self.model} with messages: {messages} and params: {self.openai_params}')
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **self.openai_params
            )
            logging.debug(f'response: {response}')
        except Exception as e:
            logging.error(f'error: {e}')
            return "error"
        return response.choices[0]['message']['content'].strip()

    def print_openai_stream(self, response):
        print ("KubeChat : ", end="")
        answer=""
        i=0
        nb=0
        for part in response:
            i+=1
            delta=part['choices'][0]['delta']
            if 'content' in delta:
                nb = nb + len(delta['content'])
                answer=answer+delta['content']
                content=delta['content']

                if i == 0:
                    content = re.sub(r"\n", "", content)

                print(content,end="")
        print()
        return answer

    def add_conversation_message(self, role, content):
        self.conversation_messages.append({"role": role, "content": content})

    def get_conversation(self):
        return self.conversation_messages

    def flatten_array(self, item_list):
        target_items = []
        for item in item_list:
            metadata = item['metadata']
            target_item = {'kind': item.get('kind')}
            for key in metadata:
                if key == 'name' or key=='namespace':
                    target_item[key] = metadata[key]
            target_items.append(target_item)
        return target_items

class ChainedCommand( KubeCommand):
    def __init__(self, sub_prompts):
        super().__init__()
        self.sub_prompts = sub_prompts

    def new_message(self, text, items, **user_params):
        self.add_conversation_message("user", text)
        response = text

        for sub_prompt in self.sub_prompts:
            answer = sub_prompt.new_message(response,items)
            response = answer.gpt_answer

        self.add_conversation_message("assistant", response)
        return  KubeChatAnswer(response, items)

class OneShotCommand( KubeCommand):
    def send_message(self):
        messages = [{'role': 'system', 'content': self.system_message}] + [ self.conversation_messages[-1] ]
        if self.stream: 
            return self.send_openai_stream(messages) 

        answer = self.send_openai_no_stream(messages)
        return answer
        return self.send_openai(messages)

class  KubeChatAnswer:
    def __init__(self, gpt_answer, items):
        self.gpt_answer = gpt_answer
        self.items = items

    # remove code block if any
    def sanitize_code_answer(self):
        self.gpt_answer = self.gpt_answer.replace('```python', '')
        self.gpt_answer = self.gpt_answer.replace('```', '')
