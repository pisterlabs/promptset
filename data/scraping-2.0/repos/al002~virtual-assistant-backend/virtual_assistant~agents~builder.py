from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents.tools import BaseTool
from langchain.agents import load_tools, ConversationalChatAgent
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# from .chat_agent import ConversationalChatAgent
from .parser import TaskOutputParser
from ..prompts.input import PREFIX, SUFFIX

from virtual_assistant.utilities import GoogleSerperAPIWrapper, PINECONE_INDEX_NAME
from virtual_assistant.tools import BrowsingTool, RetrivalTool

class AgentBuilder:
    def __init__(self, tools: list[BaseTool] = []):
        self.llm: BaseChatModel = None
        self.parser: BaseOutputParser = None
        self.global_tools: list = None
        self.tools = tools

    def build_llm(self, callback_manager: BaseCallbackManager = None):
        self.llm = ChatOpenAI(
            temperature=0, callback_manager=callback_manager, verbose=True
        )

    def build_parser(self):
        self.parser = TaskOutputParser()

    def build_global_tools(self):
        if self.llm is None:
            raise ValueError("LLM must be initialized before tools")

        toolnames = []

        embeddings = OpenAIEmbeddings()
        db = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        retriever = db.as_retriever()

        self.global_tools = [
            *load_tools(toolnames, self.llm),
            BrowsingTool(serper=GoogleSerperAPIWrapper()),
            RetrivalTool(retriver=retriever)
        ]

    def get_parser(self):
        if self.parser is None:
            raise ValueError("Parser is not initialized yet")

        return self.parser

    def get_global_tools(self):
        if self.global_tools is None:
            raise ValueError("Global tools are not initialized yet")

        return self.global_tools

    def get_agent(self):
        if self.llm is None:
            raise ValueError("LLM must be initialized before agent")

        if self.parser is None:
            raise ValueError("Parser must be initialized before agent")

        if self.global_tools is None:
            raise ValueError("Global tools must be initialized before agent")

        return ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=[
                *self.global_tools,
                *self.tools,
            ],
            system_message=PREFIX,
            human_message=SUFFIX,
            output_parser=self.parser,
        )
