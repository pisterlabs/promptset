
import time
import openai
import discord
from json import dumps, loads
from env import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class Conversations:

    openai = openai

    def __init__(self):
        self.conversations:dict[int,list[dict[str,str]]] = {}
        self.system_messages:dict[int,str] = {} # added to the conversation history before next_prompt is called

    def __str__(self) -> str:
        return str(self.conversations)
    
    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self):
        return self.conversations.__iter__()
    
    def __getitem__(self, key):
        return self.conversations[key]
    
    def __setitem__(self, key, value):
        self.conversations[key] = value
    
    def tryload(self,filename:str):
        """
        Attempts to load a conversation history from a file.
        Fails silently if the file does not exist.
        """
        try:
            with open(filename, "r") as f:
                self.conversations = loads(f.read())
        except FileNotFoundError:
            pass
    
    def save(self,filename:str):
        """
        Saves the conversation history to a file.
        """
        with open(filename, "w") as f:
            f.write(dumps(self.conversations))

    def get_history(self, member:discord.Member, stringify:bool=False):
        try:
            history = self.conversations[member.id]
            if member.id in self.system_messages:
                history.append({
                    "role": "system",
                    "content": self.system_messages[member.id]
                })
            if stringify:
                return dumps(history)
            return history
        except KeyError:
            return []

    def add_history(self, member:discord.Member, role:str, content:str):
        try:
            self.conversations[member.id].append({"role": role, "content": content})
        except KeyError:
            self.conversations[member.id] = [{"role": role, "content": content}]
    
    def next_prompt_stream(self, member:discord.Member, new_prompt:str=None):
        """
        USE FOR `CONTENT-TYPE: TEXT/EVENTSTREAM` ONLY

        Generates the next prompt for a given `member` and adds it to the current conversation.
        """
        if new_prompt is not None:
            self.add_history(member, "user", new_prompt.replace("\n"," "))
        response = None
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=self.get_history(member),
                    stream=True
                )
            except openai.error.RateLimitError:
                time.sleep(5)
        return response

    def next_prompt(self, member:discord.Member, new_prompt:str=None):
        """
        Generates the next prompt for a given `member` and adds it to the current conversation.
        """
        if new_prompt is not None:
            self.add_history(member, "user", new_prompt.replace("\n"," "))
        response = None
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=self.get_history(member)
                )
            except openai.error.RateLimitError:
                time.sleep(5)
        content:str = response.choices[0]["message"]["content"]
        self.add_history(member, "assistant", content)
        return content

    def one(self, prompt:str, system:str=None):
        prompt = prompt.replace("\n"," ")
        response = None
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.append({"role": "system", "content": system})
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages
                )
            except openai.error.RateLimitError:
                time.sleep(5)

        content:str = response.choices[0]["message"]["content"]
        return content
    
    def clear(self, member:discord.Member):
        self.conversations[member.id] = []

class shared_resource:
    """
    Used to allow the main thread to shut down subsequent threads
    """
    def __init__(self,conversations:Conversations):
        self.running = True
        self.conversations = conversations
    def stop(self):
        """
        Sets the `running` property to `False`. All tasks listening to this will stop.
        """
        self.running = False
