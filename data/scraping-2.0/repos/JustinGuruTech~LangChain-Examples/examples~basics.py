"""
This script introduces basic LangChain components. It demonstrates how to:
- Build a prompt with input variables using PromptTemplate
- Create an LLMChain, which wraps the prompt and language model for easy use
- Run the chain with input variables taken from user input
"""

from langchain import LLMChain, PromptTemplate

from config import default_llm
from utils.console_logger import ConsoleLogger, COLOR_INPUT

def main():
    # See config.py for API key setup and default LLMs
    llm = default_llm

    # Set up a PromptTemplate, which handles input variables much like an fstring. 
    # The 'input_variables' parameter lists the names of the variables that will be used in the template.
    # This building block approach is helpful as things scale
    prompt = PromptTemplate(
        input_variables=["event_objective", "team_size"],
        template="You are an event planner for corporate team-building events. Plan an event that achieves the given objective for a team of the specified size.\nEvent objective: {event_objective}\nTeam size: {team_size}"
    )

    # Create an LLMChain. This is a convenience object that wraps a prompt and an LLM,
    # providing an easy way to use the two together.
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Print start of prompt to console
    ConsoleLogger.log("Prompt:\nYou are an event planner for corporate team-building events. Plan an event that achieves the given objective for a team of the specified size.\n")

    # Get task and team size from user
    event_objective = ConsoleLogger.input_with_default(
        "Event",
        "Laser Tag" # default value
    )
    team_size = ConsoleLogger.input_with_default(
        "Team Size",
        "1 million" # default value
    )

    # Format prompt with input variables for logging
    formatted_prompt = prompt.format(
        event_objective=event_objective, 
        team_size=team_size
    )
    # Log formatted prompt
    ConsoleLogger.log_input(f"\n\n{formatted_prompt}")

    # Provide input variables to the chain's run method directly 
    # and the chain handles the template formatting
    response = llm_chain.run(
        event_objective=event_objective, 
        team_size=team_size
    ) # Thinking...


if __name__ == "__main__":
    main()
