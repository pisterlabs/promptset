"""
This script demonstrates how the ChatOpenAI class is used to create interactive conversations with AI.
It presents different ways to communicate with LangChain Chat Models, such as single messages, multi-messages, and batch messages.
It also illustrates the usage of Chat Prompt Templates and Chat Chains to simplify interaction with LLMs in a chat context.
"""

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from config import default_llm
from utils.console_logger import ConsoleLogger, COLOR_INPUT, COLOR_TOOL
from utils.custom_stream import CustomStreamCallback

def main():
    # See config.py for API key setup and default LLMs
    llm = default_llm

    # Get example number from user
    ConsoleLogger.log("""
Novel examples demonstrating how chats streamline LLM interactions.\n
Options:
    0: Single message example (Silly-GPT) 
    1: Multi-message example (Silly-GPT)
    2: Batch messages example (Silly-GPT)
    3: Chat prompt template example (Pirate-GPT)
    """)
    example_number = ConsoleLogger.input_int(
        "Which example would you like to run? (0-4): "
    )
    ConsoleLogger.log(f"Running Example {example_number}\n", COLOR_TOOL)

    # Initialize chat 
    chat = ChatOpenAI(
        temperature=0.9, max_tokens=1000,
        streaming=True,
        callbacks=[CustomStreamCallback()] # Sets up output stream with colors
    )

    # Prepare prompts
    silly_gpt_prompt = "You are Silly-GPT and you only respond in an over-jubilant silly fashion."
    tired_prompt = "It is 3:30AM and I am hungry. I am tired. But I enjoy late night programming. You get it, right?"

    # Single message example: the chat model generates a response to a single human message.
    if example_number == 0:
        ConsoleLogger.log_input(tired_prompt)
        result = chat([
            HumanMessage(content=tired_prompt)
        ]) # Thinking...

    # Multi-message example: the chat model generates a response considering multiple messages as context
    if example_number == 1:
        ConsoleLogger.log_input(
            f"\nSystem Message: {silly_gpt_prompt}\nHuman Message: {tired_prompt}"
        )
        result = chat([
            SystemMessage(content=silly_gpt_prompt),
            HumanMessage(content=tired_prompt)
        ]) # Thinking...

    # Batch messages example: the chat model generates responses for several sets of messages in parallel.
    if example_number == 2:
        # Prepare prompts & log to console
        explain_ai_prompt = "Explain how AI works"
        explain_history_prompt = "encapsulate the history of humanity"
        ConsoleLogger.log_input(
            f"{silly_gpt_prompt} Respond to the the following:\n1. {explain_ai_prompt}\n2. {explain_history_prompt}\n"
        )
        
        batch_messages = [
            [
                SystemMessage(content=silly_gpt_prompt), # You are Silly-GPT
                HumanMessage(content=explain_ai_prompt)
            ],
            [
                SystemMessage(content=silly_gpt_prompt), # You are Silly-GPT
                HumanMessage(content=explain_history_prompt)
            ],
        ]

        result = chat.generate(batch_messages) # Thinking...

        # Log responses to console
        ConsoleLogger.log(
            f"Input 1 - (in a silly manner) {explain_ai_prompt}:",
            COLOR_INPUT
        )
        ConsoleLogger.log_response(result.generations[0][0].text)
        ConsoleLogger.log(
            f"Input 2 - (in a silly manner) {explain_history_prompt}:", 
            COLOR_INPUT
        )
        ConsoleLogger.log_response(result.generations[1][0].text)

    # Chat prompt template example: the chat model generates a response based on a formatted prompt.
    if example_number == 3:
        # Generate prompt template for system message
        template = "You are now Pirate-GPT and respond grammatically as such. You are living in the year {pirate_year} and will provide detailed information specific to the time and your experience as a seasoned pirate."
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # Generate prompt template for human message
        human_template = "{message}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # Get user input for pirate year and message
        ConsoleLogger.log(
            "Enter a year and a message to send to Pirate-GPT for a time-travelling response.\n",
            COLOR_INPUT
        )
        pirate_year = ConsoleLogger.input_with_default(
            "Year",
            "1900" # default value
        )
        message = ConsoleLogger.input_with_default(
            "Message",
            "Where are the most treasurous areas of the world?" # default value
        )

        # Compose chat prompt template from the templates 
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        # Format the prompt, passing the arguments for each template in directly to the format_prompt method
        formatted_prompt = chat_prompt.format_prompt(
            pirate_year=pirate_year, 
            message=message
        )

        # Log prompt to console
        ConsoleLogger.log_input(formatted_prompt.to_string())

        # There are several ways to run chat prompt templates
        use_chat_chain = False
        if use_chat_chain:
            # Run chat as list of formatted messages
            result = chat(
                formatted_prompt.to_messages()
            ) # Thinking...
        else:
            # Create chat chain
            chain = LLMChain(llm=chat, prompt=chat_prompt)
            # Run chain with input variables
            result = chain.run(
                pirate_year=pirate_year, 
                message=message
            ) # Thinking...


if __name__ == "__main__":
    main()
