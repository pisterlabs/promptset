# An example of using ChatOpenAI with a HumanMessage:

# An example of using ChatOpenAI with a HumanMessage: In this section, we are trying to use the LangChain library to create a chatbot that can translate an English sentence into French. This particular use case goes beyond what we covered in the previous lesson. We'll be employing multiple message types to differentiate between users' queries and system instructions instead of relying on a single prompt. This approach will enhance the model's comprehension of the given requirements.

# First, we create a list of messages, starting with a `SystemMessage` that sets the context for the chatbot, informing it that its role is to be a helpful assistant translating English to French. We then follow it with a `HumanMessage` containing the user’s query, like an English sentence to be translated.

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    openai_api_key=apikey,
    model="gpt-4",
    temperature=0
)

message = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
]

# chat(message)


# =================================================================================

batch_messages = [
  [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate the following sentence: I love programming.")
  ],
  [
    SystemMessage(content="You are a helpful assistant that translates French to English."),
    HumanMessage(content="Translate the following sentence: J'aime la programmation.")
  ],
]
print( chat.generate(batch_messages) )
