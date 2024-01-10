import os
import json
import time
from pyexpat.errors import messages
from openai import OpenAI
import datetime
from langchain.agents.openai_assistant import OpenAIAssistantRunnable

assistant = OpenAIAssistantRunnable.create_assistant(
    name="Test Assistant",
    instructions="You are a very helpful assistant",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)

output = assistant.invoke({"content": "Hello, my name is Davide"})

for message in output:
    #print(message.content[0].text.value)
    print(message.content)

output = assistant.invoke({"content": "What's my name again?"})

for message in output:
    print(message.content)