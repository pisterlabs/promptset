from typing import Callable, Dict, List, Optional, Union
from langchain.agents import load_tools, initialize_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
import pinecone
from chatbot_settings import ChatBotSettings
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from chatbot_settings import ChatBotSettings
from bot_abstract_class import BotAbstract



class BotAgentTools(BotAbstract):
    def __init__(self, chatBotSettings: ChatBotSettings()):
        self.chatBotSettings = chatBotSettings

        self.llm = chatBotSettings.llm
        self.memory = chatBotSettings.memory
        self.tools = chatBotSettings.tools


    def get_bot_response(self, message):
        tool_names = self.tools
        tools = load_tools(tool_names)

        agent = initialize_agent(
            tools, self.llm, agent="zero-shot-react-description", verbose=True)

        response = agent.run(message)

        return response



    
    
