from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from actor.agents.rest_agent import run_rest_agent

model_main_agent = "gpt-4"
model_rest_agent = "gpt-3.5-turbo"


def run_main_agent() -> str:
    prefix = f"""
        Pretend you are a person playing a game.
        You are on a 2D-grid and can see on the current and directly to adjacent cells.
        Goal of the game: You need to find and pick up the treasure. You also know the distance to the treasure.
        
        Your actions: Send commands to reach the goal.

        Rules of the game:
        - Cell descriptions: W=wall, G=ground, T=treasure
        - Available commands (choose freely): START UP DOWN LEFT RIGHT PICK_UP
        """
    agent = initialize_agent(
        tools=[
            Tool(
                name="RestAgent",
                func=run_rest_agent,
                description="""useful when you need to call the api to send the command and return the content of the response
                DO NOT SEND THE ACTUAL REQUEST TO THIS TOOL. JUST SEND THE COMMAND.""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model=model_main_agent),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        verbose=True,
    )
    return agent.run(prefix + "\nStart the game and find the treasure")