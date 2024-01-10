import abc
import os
import time

import openai
import tiktoken


class ChatHelper(abc.ABC):
    @abc.abstractmethod
    def chat(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def create_summary(self):
        raise NotImplementedError
    
class OpenAI_BaseChatHelper(ChatHelper):
    def __init__(self, client=None):
        self.client = client
        if not self.client:
            self.client = openai.OpenAI()

        self.model = "gpt-3.5-turbo-1106"

        with open(os.path.join(os.path.dirname(__file__), "config/prompts/system.txt"), "r") as fopen:
            self.system_prompt = fopen.read()

        with open(os.path.join(os.path.dirname(__file__), "config/prompts/summary.txt"), "r") as fopen:
            self.summary_prompt = fopen.read()  
    
class OpenAI_AssistantHelper(OpenAI_BaseChatHelper):
    def __init__(self, client=None):
        super().__init__(client)
        assistants = [x for x in self.client.beta.assistants.list() if x.name == "Robit"]

        if len(assistants) == 0:
            self.assistant = self.client.beta.assistants.create(
                name="Robit",
                instructions=self.system_prompt,
                model=self.model
            )

        elif len(assistants) == 1:
            self.assistant = assistants[0]
        else:
            # TODO: logger warn, multiple assistants
            pass


        # TODO: Logic for creating and managing new threads
        self.thread = self.client.beta.threads.create()

        # TODO: handle all states appropriately
        self.done_states = ["cancelled", "failed", "completed", "expired"]

    def chat(self, message:str, run_instructions:str=None):
        thread = self.thread
            
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
            instructions=run_instructions
        )

        while run.status not in self.done_states:
            # TODO: Smarter wait logic
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        messages = self.client.beta.threads.messages.list(thread_id=thread.id).data
        # TODO: Parse messages and return appropriate messages and/or convert to self.get_messages()
        return messages[0].content[0].text.value
    
    def create_summary(self):
        return self.chat(message=self.summary_prompt, run_instructions="This user is an administrator. Provide what they ask to the best of your ability.")
    
    def get_messages(self):
        # TODO: Parse these a bit and make a more extensible object (to use in chat())
        try:
            return [x.content[0].text.value for x in self.client.beta.threads.messages.list(thread_id=self.thread.id).data]
        except AttributeError:
            return []


class OpenAI_ChatHelper(OpenAI_BaseChatHelper):
    def __init__(self, client=None):
        super().__init__(client)
        self.history = []
        self.max_response_tokens = 15

        self.history.append({"role": "system", "content": self.system_prompt})

    def get_robit_message(self, messages:list):
        if not messages:
            messages = self.history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.get_total_allowed_response_tokens(messages)
        )
        
        return response.choices[0].message.content
    
    def chat(self, message:str):
        self.history.append({
            "role": "user", "content": message
        })
        response = self.get_robit_message(self.history)
        self.history.append({
            "role": "assistant", "content": response
        })
        return response

    def get_total_allowed_response_tokens(self, messages:dict):
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = 0
        for message in messages:
            num_tokens += 3
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += 1
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens + self.max_response_tokens
    
    def create_summary(self):
        return self.get_robit_message(self.history + [{"role":"user", "content":self.summary_prompt}])