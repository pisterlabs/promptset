"""
This script introduces the use of agents in LangChain.
Agents can determine when a tool is necessary for a task.
For example, determining when a web search or a calculator is needed.
It shows examples of setting up an Agent with tools and parameters, then running the Agent to achieve specific tasks.
"""

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from config import default_llm
from utils.console_logger import ConsoleLogger, COLOR_INPUT, COLOR_ERROR

def main(): 
    # See config.py for API key setup and default LLMs
    llm = default_llm

    # Get example number from user
    ConsoleLogger.log("""
Examples showing how agents can use tools & maintain memory.\n
Options:
    0: Web search (Requires Serpapi API key)
    1: Web search & calculator (Requires Serpapi API key)
    2: ConversationBufferMemory Agent for context retention
    """)
    example_number = ConsoleLogger.input_int(
        "Which example would you like to run? (0-2): "
    )
    ConsoleLogger.log(f"Running Example {example_number}\n", COLOR_INPUT)

    if (example_number == 0):
        # Serpapi enables web search
        tools = load_tools(["serpapi"])

        # Set up agent with tools & parameters
        agent = initialize_agent(
            tools, 
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            verbose=True # This logs the agent's actions to the console
        )

        # Get company name from user
        ConsoleLogger.log(
            "This example asks for a recent article, which indicates to the Agent it should use it's web searching tool.", 
            COLOR_INPUT
        )
        company_name = ConsoleLogger.input_with_default(
            "Company Name",
            "Guru Technologies", # default value
        )

        # Create prompt for agent and log to console
        agent_prompt = f"Find a recent article on {company_name}, summarize it's content, and describe what the company does."
        ConsoleLogger.log_input(agent_prompt)

        agent.run(agent_prompt) # Thinking...

    if (example_number == 1):
        # Load tools for the agent. Some tools like llm-math require a LLM as an arg.
        tools = load_tools(["serpapi", "llm-math"], llm=llm)

        # Initialize an agent with the tools, language model, and type of agent
        agent = initialize_agent(
            tools, 
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # Simple response
            verbose=True
        )

        # Create prompt for agent and log to console
        agent_prompt = "Find a ridiculously long PEMDAS math problem online and solve it."
        ConsoleLogger.log_input(agent_prompt)

        # Run agent
        agent.run(agent_prompt) # Thinking...

    if (example_number == 2):
        # Define a custom prompt for the conversation
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Set up buffer memory for the conversation, which will retain the last few messages
        memory = ConversationBufferMemory(return_messages=True)

        # Initialize ConversationChain with custom prompt, memory, and language model
        conversation = ConversationChain(
            memory=memory, 
            prompt=prompt, 
            llm=llm,
            verbose=True # Context is logged to console (in green)
        )

        # Interact with the conversation agent
        input = "Exactly what year did humans first develop consciousness?"
        ConsoleLogger.log_input(input)
        response = conversation.predict(input=input) # Thinking...

        input = "I see. In that case, what are some of the earliest signs of consciousness?"
        ConsoleLogger.log_input(input)
        response = conversation.predict(input=input) # Thinking...

        input = "Can you summarize this conversation in 3 bullet points?"
        ConsoleLogger.log_input(input)
        response = conversation.predict(input=input) # Thinking...

    else:
        ConsoleLogger.log("Invalid example number. Exiting...", COLOR_ERROR)

if __name__ == "__main__":
    main()
