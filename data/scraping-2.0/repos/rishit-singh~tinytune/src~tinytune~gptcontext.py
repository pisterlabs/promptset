import openai
import json
from tinytune.util.prompt import ValidatePrompt
from typing import Callable, Any

class GPTContext:    
    def __init__(self, model: str, apiKey: str, promptFile: str = None):
        self.Model: str = model
        self.APIKey: str = apiKey
        self.Messages: list[dict[str]] = []
        openai.api_key = self.APIKey

    def LoadMessages(self, promptFile: str = "prompts.json"):
        self.PromptFile = promptFile

        with open(promptFile, "r") as fp:
            self.Messages = json.load(fp)

    def Save(self, promptFile: str = "prompts.json") -> bool: 
        try:
            with open(promptFile, "w") as fp:
                json.dump(self.Messages, fp, indent=2)
                
        except:
            print("An error occured in saving messages.")
            return False

        return True
    
    def AddPrompt(self, prompt: dict, onError: Callable[[KeyError], Any] = None) -> bool:
        try:
            ValidatePrompt(prompt)
            self.Messages.append(prompt)

        except KeyError as e:
            if (onError == None):
                raise e
            onError(e)

            return False

        return True

    def Send(self, _messages: list[dict[str, str]]) -> dict:
        print(f"Message size: {len(self.Messages)}")
        print(_messages)

        if (len(self.Messages) < 1):
            for message in _messages:
                self.Messages.append(message)

        return dict(openai.ChatCompletion.create(model=self.Model, messages=_messages)["choices"][0]["message"])

    def Prompt(self, role: str, message: str) -> list[dict[str, str]]:
        self.Messages.append(dict({ "role": role, "content": message }))
        self.Messages.append(self.Send(self.Messages))

        return self.Messages
    
    def Update(self):
        self.Send(self.Messages)
