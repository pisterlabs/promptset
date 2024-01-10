from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv, find_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

''' 
Experiment with the web search tool to see if it is better than what we have built
'''

_ = load_dotenv(find_dotenv()) # read local .env file


# Vectorstore
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(),persist_directory="./chroma_db_oai")

# LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Search
search = GoogleSearchAPIWrapper()

# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)

user_input = "What information around budget and sustainability do you have for City Of Orlando? and you break down your reasoning step by step?"
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever)
result = qa_chain({"question": user_input})

print(result)