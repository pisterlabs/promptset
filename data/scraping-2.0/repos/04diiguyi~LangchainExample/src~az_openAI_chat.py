# To run: In the current folder: 
# python az_openAI_chat.py

# This example is a sample that uses OpenAI gpt3.5 as a chat model
# It includes three types of requests, 
# The first is a chat with only human messages
# The second is a chat with both system and human messages
# The third one is for streaming chat

# Example response
# J'aime programmer.

# There were several important events that happened in 1986, 
# but one of the most significant events was the explosion of the Chernobyl 
# nuclear power plant in Ukraine on April 26, 1986. It was the worst nuclear 
# disaster in history, and it had a significant impact on the environment and 
# the health of people in the surrounding areas.

# Verse 1:
# Bubbles rising to the top
# A refreshing drink that never stops
# Clear and crisp, it's oh so pure
# Sparkling water, I can't ignore
# ...

import os

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

chat = AzureChatOpenAI(deployment_name=Az_Open_Deployment_name_gpt35,
            openai_api_version="2023-03-15-preview", temperature=0)

response = chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

print(response.content)

messages = [
    SystemMessage(content="You are a helpful assistant that helps user to find information."),
    HumanMessage(content="What is the most important event happened in 1986?")
]

response = chat(messages)

print(response.content)

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat_stream = AzureChatOpenAI(deployment_name=Az_Open_Deployment_name_gpt35,
            openai_api_version="2023-03-15-preview", 
            streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat_stream([HumanMessage(content="Write me a song about sparkling water.")])

print(resp)