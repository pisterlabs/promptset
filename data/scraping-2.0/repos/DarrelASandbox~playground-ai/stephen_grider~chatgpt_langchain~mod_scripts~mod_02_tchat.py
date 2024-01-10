"""
A simple chat application using OpenAI's language models.

This script demonstrates the creation of a conversational AI application using the langchain library.
It utilizes OpenAI's GPT-based models to generate responses in a chat-like interface.
Key features include loading environment variables, setting up a chat model with memory,
and processing input through a conversational chain.

Features:
- Load environment variables from a .env file for configuration.
- Use ChatOpenAI to interact with OpenAI's language models.
- Employ ConversationBufferMemory to maintain a history of the conversation, storing it in 'messages.json'.
- Construct a ChatPromptTemplate to format the chat input and history for the language model.
- Use LLMChain to process input by combining the model, prompt template, and memory.
- A loop to continuously accept user input and print responses from the chat model.

The script is intended for those new to language models but familiar with Python programming.

Usage:
Run the script and interact with the chatbot through the command line.
Type your message after the prompt '>> ' and receive a response from the AI model.
"""
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


load_dotenv()


# Initialize a ChatOpenAI instance.
# This creates an object to interact with OpenAI's language models.
chat = ChatOpenAI()

# Set up memory for the conversation.
# Here, we use ConversationBufferMemory to maintain a buffer of conversation history.
# The history is stored in a file named 'messages.json'.
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True,
)

# Create a prompt template for the language model.
# This template defines how the input ("content") and conversation history ("messages")
# are formatted before being sent to the language model.
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# Initialize the LLMChain.
# This chain combines the language model, prompt, and memory components to process input.
chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

# Main loop for the chat application.
# This continuously accepts user input, processes it through the chain, and outputs the result.
while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result)
