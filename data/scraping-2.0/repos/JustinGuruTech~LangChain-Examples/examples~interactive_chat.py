"""
This script demonstrates a simple interactive conversation loop with a user.
It uses LangChain components to handle conversation management, memory, 
and language model interactions.
"""

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from config import default_llm
from utils.console_logger import ConsoleLogger, COLOR_INPUT

def main():
    # See config.py for API key setup and default LLMs
    llm = default_llm

    # Prepare the Chat Prompt Template that will be passed to the ConversationChain.
    # It includes a system message, a placeholder for the conversation history,
    # and a human message for user input.
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful and friendly AI, trained to answer questions about the world and the creatures, places, and things that inhabit it."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Set up simple memory for the conversation
    memory = ConversationBufferMemory(return_messages=True)

    # Wrap everything in a ConversationChain
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    # Interactive chat loop
    print("Welcome to the interactive chat! Type 'exit' to end the conversation.", COLOR_INPUT)
    while True:
        # Get user input
        user_input = ConsoleLogger.input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Run the conversation with user input
        response = conversation.predict(input=user_input) # Thinking...


if __name__ == "__main__":
    main()