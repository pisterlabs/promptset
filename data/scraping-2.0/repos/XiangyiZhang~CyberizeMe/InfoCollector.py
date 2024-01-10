from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain import PromptTemplate
import asyncio

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from prompt_templates import info_collector_template
from CustomTools import get_characteristics, get_status, get_relationship, get_thoughts

# llm = ChatOpenAI(openai_api_key=openai_key, temperature=0.1, model="gpt-3.5-turbo-0613")

class InfoCollector():
    '''
        The information collector to collect relevant information to reply the message.
        Collects a report about the persona to help downstream LLM better generate reply.
    '''

    def __init__(self,llm,agent_name:str, sender:str):
        self.llm = llm
        self.agent_name = agent_name
        self.sender = sender
        self.tools = [get_characteristics(), 
                      get_status(), 
                      get_relationship(), 
                      get_thoughts()]

    # Collect relevant info by starting an OpenAI Functions Agent
    async def collect(self, message:str) -> str:
        info_collector = initialize_agent(self.tools, self.llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

        info_collector_prompt = PromptTemplate(
            input_variables=["agent_name","sender","message",],
            template=info_collector_template,
        )

        report = await info_collector.arun(info_collector_prompt.format(agent_name=self.agent_name, 
                                                                        sender=self.sender, 
                                                                        message=message))
        


        return report