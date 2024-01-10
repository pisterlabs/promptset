from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from llms.azure_llms import create_llm
from tools.get_tools import qa_tools
from langchain.chains import ConversationalRetrievalChain
from tools.kendra.retriever import KendraRetriever

llm = create_llm()

llm.request_timeout = 15

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3, return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=KendraRetriever(), memory=memory, verbose=True)