import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()

threadId = open("threads.txt", "r").read()
# solFile = open("PasswordStore.sol", "r").read()

message = client.beta.threads.messages.create(
    thread_id=threadId,
    role="user",
    content="/analyseContract"
)

print("Adding message to thread...")
