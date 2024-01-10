from langchain import OpenAI
from src.utils.common import read_yaml

from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType
from langchain.agents import initialize_agent
#from langchain.tools import Tool 
from langchain.agents import Tool
from src.custom_toolkit import WeatherTool, awsEC2Tool, DummyTool

config = read_yaml("src/config.yaml")

MODEL_NAME = config['chatbot']['MODEL_NAME']
TEMPERATURE = config['chatbot']['TEMPERATURE']


llm = OpenAI(
    model_name=MODEL_NAME,
    temperature=TEMPERATURE
)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

def get_conv_agent():
    tools = [awsEC2Tool(), WeatherTool()]
    tools.append(
        Tool.from_function(
            func=search.run,
            name="Search",
            description="Only Use this tool only when you need to answer something about the recent or current envent"
            ))
    # class CalculatorInput(BaseModel):
    #     question: str = Field(description = "The input string with the number being computed")
    #tools.append(WeatherTool())
    tools.append(
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="useful for when you need to answer questions about math",
            #args_schema=CalculatorInput
            # coroutine= ... <- you can specify an async method if desired as well
            ))
    tools.append(DummyTool())
    
    #tools.append(awsEC2Tool())
    
    conv_agent = initialize_agent(
                                agent= AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                tools=tools,
                                llm=llm,
                                verbose=True,
                                max_iterations=3,
                                early_stopping_method='generate',
                                #memory=ConversationSummaryMemory(llm=llm),
                                handle_parsing_errors="The chain should end with the final result",
                                )
    conv_agent.agent.llm_chain.prompt.messages[0].prompt.template = f"""
        Introduce yourself as an AI Chatbot developed by Hasanain. but don't label it as "System:"\n
        {str(conv_agent.agent.llm_chain.prompt.messages[0].prompt.template)}\n
        Don't use the Search tool until there is a need to search something online, something which has happened recently.\n
        The chain should end with a single string, that means the final answer should be in a string format.
        Remove the "Thought :" at the end of the chain, only provide the final answer!
        """
    return conv_agent