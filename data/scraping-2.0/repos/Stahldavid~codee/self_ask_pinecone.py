from langchain import OpenAI, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pinecone_vector import PineconeVector
from dotenv import load_dotenv
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
import os
import pinecone


# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(temperature=0)


pinecone.init(
            api_key=pinecone_api_key,  # find at app.pinecone.io
            environment="us-east1-gcp"  # next to api key in console
        )

index1 = pinecone.Index("langchain-chat")

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Initialize a Pinecone vector store
vectorstore1 = Pinecone(index=index1, embedding_function=embeddings.embed_query, text_key="text")

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
vectorstore_info = VectorStoreInfo(
    name="Robotics and Control Systems Knowledge Base",
    description="A collection of information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller",
    vectorstore=vectorstore1,
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# Create a RetrievalQA chain using the Pinecone vector store
pinecone_search_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore1.as_retriever())

tools = [
    Tool.from_function(
        func=pinecone_search_chain.run,
        name = "Intermediate Answer",
        description="Useful for search information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller"
        # coroutine= ... <- you can specify an async method if desired as well
    ),
]



self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("What is The best method for variable Impedance control for force feedback and interaction with virtual envirolment?")