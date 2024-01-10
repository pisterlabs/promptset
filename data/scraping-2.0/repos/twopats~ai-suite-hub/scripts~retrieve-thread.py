
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()

threadId = open("threads.txt", "r").read()

messages = client.beta.threads.messages.list(thread_id=threadId)
print(messages)
