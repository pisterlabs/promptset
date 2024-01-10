from dotenv import load_dotenv
from langchain.agents import AgentType, create_pandas_dataframe_agent, initialize_agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool, Tool
from pandas import DataFrame

model_python_agent_executor = "gpt-3.5-turbo"
model_pandas_agent = "gpt-3.5-turbo"
model_grand_agent = "gpt-3.5-turbo"


def run(
    prompt: str, df: DataFrame, st_callback: StreamlitCallbackHandler, openai_api_key:str, model="gpt-4"
) -> str:
    agent_executor_kwargs = {
        "handle_parsing_errors": True,
    }
    pandas_agent = create_pandas_dataframe_agent(
        llm=ChatOpenAI(
            temperature=0,
            model=model_pandas_agent,
            streaming=True,
            openai_api_key=openai_api_key
        ),
        df=df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs=agent_executor_kwargs,
        max_iterations=5
    )

    # currently disabled, agent that can call other agent
    grand_agent = initialize_agent(
        tools=[
    #         Tool(
    #             name="PythonAgent",
    #             func=python_agent_executor.run,
    #             description="""useful when you need to transform natural language and write from it python and execute the python code,
    # returning the results of the code execution,
    # DO NOT SEND PYTHON CODE TO THIS TOOL""",
    #         ),
            Tool(
                name="PandasAgent",
                func=pandas_agent.run,
                description="""useful when you need to answer question for a provided pandas dataframe, This tool already knows which dataframe to handle.
                                               takes as an input the entire question and returns the answer after running calculations""",
            ),
        ],
        llm=ChatOpenAI(temperature=0, model=model_grand_agent, streaming=True, openai_api_key=openai_api_key),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    response = pandas_agent.run(prompt, callbacks=[st_callback])
    return response
