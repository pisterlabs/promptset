import tkinter as tk
import importlib.util
import json

import openai

from .gui import Settings, Conversation
from .utils import tokens2price

class FuncGPT:
    def __init__(
            self,
            kanu,
            openai_key,
            model,
            temperature,
            prompt,
            function_script
    ):
        self.kanu = kanu
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.function_script = function_script
        openai.api_key = openai_key
        self.settings = Settings(self)
        self.conversation = Conversation(self)
        self.tokens = 0
        self.price = 0
        self.module = importlib.machinery.SourceFileLoader("", self.function_script).load_module()

    def run(self):
        self.conversation.page()

    def send_message(self):
        if not self.messages:
            self.messages.append({"role": "system", "content": self.prompt})
        self.messages += [{"role": "user", "content": self.user_input.get()}]
        try:
            first_response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                functions=[x["json"] for x in self.module.functions.values()],
                function_call="auto",
            )
            message = first_response["choices"][0]["message"]
            if message.get("function_call"):
                function_name = message["function_call"]["name"]
                function_args = json.loads(message["function_call"]["arguments"])
                function_response = self.module.functions[function_name]["function"](**function_args)
                second_response = openai.ChatCompletion.create(
                    temperature=self.temperature,
                    model=self.model,
                    messages=[
                        {"role": "user", "content": self.user_input.get()},
                        message,
                        {"role": "function", "name": function_name, "content": function_response},
                    ],
                )
                self.calculate_usage(second_response, function=function_name)
                answer = second_response["choices"][0]["message"]["content"]
            else:
                answer = first_response["choices"][0]["message"]["content"]
            self.messages += [{"role": "assistant", "content": answer}]
            self.session.insert(tk.END, "You: " + self.user_input.get() + "\n", "user")
            self.session.insert(tk.END, f"Bot: " + answer + "\n", "bot")
            self.calculate_usage(first_response)
            self.chatbox.delete(0, tk.END)
        except openai.error.InvalidRequestError as e:
            error = str(e)
            if "Please reduce the length of the messages or functions." in error:
                self.system.insert(tk.END, f"System: {error} You can also create a new chat session.\n", "system")
                self.messages.pop()
            else:
                raise

    def calculate_usage(self, response, function=None):
        total_tokens = response["usage"]["total_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        prompt_price = tokens2price(self.model, "prompt", prompt_tokens)
        completion_price = tokens2price(self.model, "completion", completion_tokens)
        self.price += prompt_price + completion_price
        self.tokens += total_tokens
        if function is None:
            message = f"System: Used {prompt_tokens:,} prompt + {completion_tokens:,} completion = {total_tokens:,} tokens (total: {self.tokens:,} or ${self.price:.6f})."
        else:
            message = f"System: Used {prompt_tokens:,} prompt + {completion_tokens:,} completion = {total_tokens:,} tokens (total: {self.tokens:,} or ${self.price:.6f}) (called function: {function})."   
        self.system.insert(tk.END, f"{message}\n", "system")

    def clear_session(self):
        self.tokens = self.price = 0
        self.run()