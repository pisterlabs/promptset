from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# iterate over image assets and transform them given the prompt
def game_visual_updater(prompt):
    pass
    

# return a 2D array consistenting of elements that can be used to create a Mario level
def level_creator(prompt):
    pass


tools = [
    Tool(
        name = "Update Game",
        func=game_visual_updater,
        description="useful for when you need to update the visual details of the video game. You should define the ways in which you need to update the game."
    ),
    Tool(
        name="Create Level",
        func=level_creator,
        description="useful for when you need to create a new level for the video game."
    ),
]

main_agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

def response_function(prompt, history):
    # main_agent.run(prompt)
    return "Hi."