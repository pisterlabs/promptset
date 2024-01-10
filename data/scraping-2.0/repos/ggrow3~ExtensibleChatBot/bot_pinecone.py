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
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from chatbot_settings import ChatBotSettings
from bot_abstract_class import BotAbstract



class BotPineCone(BotAbstract):
    def __init__(self, chatBotSettings: ChatBotSettings()):
        self.chatbotSettings = chatBotSettings

        self.llm = chatBotSettings.llm
        self.memory = chatBotSettings.memory
        self.index_name = ChatBotSettings().PINECONE_INDEX()

    def get_bot_response(self, message):
        pinecone.init(
            api_key=ChatBotSettings().PINECONE_API_KEY(),
            environment=ChatBotSettings().PINECONE_API_ENV()
        )
       
        embeddings = OpenAIEmbeddings(openai_api_key=ChatBotSettings().OPENAI_API_KEY())
        pine = Pinecone.from_existing_index(self.index_name, embeddings)

        chain = load_qa_chain(self.llm, chain_type="stuff")

        docs = pine.similarity_search(message, include_metadata=True)
        response = chain.run(input_documents=docs, question=message)
        docs = pine.similarity_search(message, include_metadata=True)
        response = chain.run(input_documents=docs, question=message)

        # If there is no matching response, provide a default response
        return response



    
