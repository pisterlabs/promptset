from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.graphs import Neo4jGraph

from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(
    url="neo4j://ec2-107-22-100-129.compute-1.amazonaws.com:7687/", 
    username="mingran", 
    password="mingran123")


class MovieAgent(AgentExecutor):
    """Movie agent"""

    @staticmethod
    def function_name():
        return "MovieAgent"
    
    def return_values(self) -> list[str]:
        """Return values of the agent."""
        return ["output"]

    @classmethod
    def initialize(cls, movie_graph, model_name, *args, **kwargs):
        if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            llm = ChatOpenAI(temperature=0, model_name=model_name)
        else:
            raise Exception(f"Model {model_name} is currently not supported")

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
        readonlymemory = ReadOnlySharedMemory(memory=memory)

        agent_chain =  GraphCypherQAChain.from_llm(ChatOpenAI(temperature=0), graph=graph, verbose=True, stop=["Output:"])
        

        # Load the tool configs that are needed.
        # tools = [
        #     Tool(
        #         name="Cypher search",
        #         func=cypher_tool.run,
        #         description="""
        #         Utilize this tool to search within a gene knowledge graph database, specifically designed to answer gene, disease and drug-related questions.
        #         This specialized tool offers streamlined search capabilities to help you find the gene information you need with ease.
        #         Input should be full question.""",
        #     ),
            
            
        # ]

        # agent_chain = initialize_agent(
        #     tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
        

        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)

        return result
