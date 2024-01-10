from abc import ABC, abstractmethod
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from modules.functions.create_prompt import create_prompt
from modules.roles_templates.q_a_template import (
    human_q_a_template,
    system_q_a_template,
)


from dotenv import load_dotenv

load_dotenv()


class ChainGeneral(ABC):
    @abstractmethod
    def execute_chain(self):
        pass


class MeetingChain(ChainGeneral):
    def __init__(self):
        pass

    def execute_chain(self):
        return "MeetingChain"


class Q_AChain(ChainGeneral):
    def __init__(self):
        self.__llm = OpenAI(temperature=0)
        tools = load_tools(["serpapi", "llm-math"], llm=self.__llm)
        self.__agent = initialize_agent(
            tools,
            llm=self.__llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def execute_chain(self, request: str):
        functions_response = self.__agent.run(request)
        prompt = create_prompt(
            system_prompt=system_q_a_template,
            human_prompt=human_q_a_template,
            input_variables=["output"],
        )
        functions_chain = prompt | self.__llm | StrOutputParser()
        response = functions_chain.invoke({"output": functions_response})
        return response
