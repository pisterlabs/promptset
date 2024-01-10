import os
from dotenv import load_dotenv

import openai

# Load environment variables from .env file
load_dotenv("/Users/adam/PycharmProjects/LLM_api/.env")

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatAgent:
    def __init__(self, name, system_prompt="You are a helpful assistant."):
        self.name = name
        self.system_prompt = f"You are called: {self.name} " + system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def send_message(self, content):
        self.messages.append({"role": "user", "content": content})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        assistant_message = completion.choices[0].message["content"]
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def converse_with(self, other_agent, content):
        response = other_agent.send_message(content)
        print(f"{other_agent.name}:\n {response}")
        return response if response else "END"


if __name__ == "__main__":
    alice = ChatAgent(
        "Alice",
        """
        You are a drunken sailor from the 1800s. You are trying to hide that you are drunk. 
        
        Stay in character!
        If it is clear the conversation has ended respond with 'END'.
        """
    )
    bob = ChatAgent(
        "Bob",
        """
        You are an admiral that only speaks in short sentences.
        
        Stay in character!
        If it is clear the conversation has ended respond with 'END'.
        """
    )

    # Example conversation
    # Start the conversation
    message = "Ow (you brush past)"
    print(f"Alice:\n {message}")
    for _ in range(20):  # Let's limit the conversation to 5 exchanges for this example
        message = alice.converse_with(bob, message)
        if message.startswith("END") or message.endswith("END.") or "END" in message:
            break
        message = bob.converse_with(alice, message)
        if message.startswith("END") or message.endswith("END.") or "END" in message:
            break
