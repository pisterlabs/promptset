import openai
import re

openai.api_key = "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"


class BotFriendPreminium():
    def __init__(self):
        
        self.prompt = [
        ]

    def api_query(self,inpt):
        self.prompt.append({"role": "user", "content": inpt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.prompt,
            max_tokens=150
        )

        system_message = response["choices"][0]["message"]["content"]
        self.prompt.append({"role": "assistant", "content": system_message})

        return system_message
    
    def add_prompt(self, text):
        self.prompt.append({"role": "system", "content":text})
        print("promt ",text)