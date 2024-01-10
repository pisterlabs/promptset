# agents.py
from openai import OpenAI

class GeneratorAgent:
    def __init__(self, model:str, system_init_content:str):
        self.model = model
        self.content_history = []
        self.system_init_content = system_init_content
        self.content_history.append({
            "role": "system",
            "content": system_init_content
        })

    def __len__(self):
        return len(self.content_history)


    def __call__(self, input_prompt:str):
        prompt = {
            "role": "user",
            "content": input_prompt
        }
        self.content_history.append(prompt)
        response = self.__chat_completion()
        self.content_history.append(response)
        content = response.content.strip()
        return content

    def __chat_completion(self):
        client = OpenAI()
        completion = client.chat.completions.create(
            model = self.model,
            messages = self.content_history
        )
        response = completion.choices[0].message
        return response

    def retry_completion(self):
        self.content_history = self.content_history[:-1]
        content = self.__chat_completion()
        return content

    def undo_completion(self):
        self.content_history = self.content_history[:-2]

    def clear_completion(self):
        self.content_history = self.system_init_content

class InterpreterAgent:
    def __init__(self, model:str, system_init_content:str):
        self.model = model
        self.content_history = []
        self.system_init_content = system_init_content
        self.content_history.append({
            "role": "system",
            "content": system_init_content
        })

    def __len__(self):
        return len(self.content_history)


    def __call__(self, input_prompt:str):
        prompt = {
            "role": "user",
            "content": input_prompt
        }
        self.content_history.append(prompt)
        response = self.__chat_completion()
        self.content_history.append(response)
        content = response.content.strip()
        return content

    def __chat_completion(self):
        client = OpenAI()
        completion = client.chat.completions.create(
            model = self.model,
            messages = self.content_history,
            response_format={ "type": "json_object" }
        )
        response = completion.choices[0].message
        return response

    def retry_completion(self):
        self.content_history = self.content_history[:-1]
        content = self.__chat_completion()
        return content

    def undo_completion(self):
        self.content_history = self.content_history[:-2]

    def clear_completion(self):
        self.content_history = self.system_init_content

class ValidatorAgent:
    def __init__(self, model:str, system_init_content:str):
        self.model = model
        self.system_init_content = system_init_content
        self.content_history = []
        self.content_history.append({
            "role": "system",
            "content": system_init_content
        })

    def __call__(self, input_prompt:str):
        prompt = {
            "role": "user",
            "content": input_prompt
        }
        self.content_history.append(prompt)
        response = self.__chat_completion()
        content = response.content.strip()
        return content

    def __chat_completion(self):
        client = OpenAI()
        completion = client.chat.completions.create(
            model = self.model,
            messages = self.content_history,
            response_format={ "type": "json_object" }
        )
        response = completion.choices[0].message
        return response