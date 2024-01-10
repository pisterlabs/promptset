import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.agents import initialize_agent, AgentType, Tool

from backend import config
from backend.resume_generation import ResumeGenerator

class CustomAgent():
    """
    Agent to control the conversation with tools
    """
    def __init__(self):
        os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
        self.llm = OpenAI(temperature=0, max_tokens=1200)
        # set the two memory
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        readonlymemory = ReadOnlySharedMemory(memory=self.memory)
        
        # build the resume generator
        self.resume_generator = ResumeGenerator(self.llm, readonlymemory)
        # allow the model to choose from two actions
        tools = [
            Tool(
                name = "Update Prompts",
                func=self.resume_generator.run,
                description="useful for when you need to re-generate the resume based on user request. Users may have special requirements and preferences and you will need to use this function to update the resume."
            ),
            Tool(
                name = "Generate Resume",
                func=self.resume_generator.run,
                description="useful for when you need to generate resume and the user does not have specific requirements. If the user have no new instructions, you will need to use this function to generate a new resume."
            )
        ]

        self.agent = initialize_agent(tools, self.llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=self.memory)
    
    def run(self, query):
        """
        Run the agent to make responses
        """
        return self.agent.run(query)