"""
agent.service.py
"""
import os
import openai
import pinecone

from langchain.agents import Tool, initialize_agent
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("OPENAI_API_BASE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

INDEX_NAME = "hackaton-anthropic"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

# Validate openAI azure
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT

EMBEDDINGS_MODEL = "text-embedding-ada-002"

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(INDEX_NAME)

# embeddings
embeddings = OpenAIEmbeddings(deployment_id=EMBEDDINGS_MODEL, 
                                chunk_size=1,
                                openai_api_key=API_KEY,
                                openai_api_base=RESOURCE_ENDPOINT
                                )

llm_claude = ChatAnthropic(model_name="claude-2", 
                           max_tokens=2000, 
                           anthropic_api_key=ANTHROPIC_API_KEY, 
                           temperature=0.0)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
        )

# vectorstore
vectorstore = Pinecone(index, embeddings.embed_query, "text")


# qa tool
qa = RetrievalQA.from_chain_type(
        llm=llm_claude,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
        )

# Tools
tools = [
    Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            'Always use this tool to solve any question a user have. It will provide you the context you need to reply to the user.'
        )
    )
]

# Agent
agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm_claude,
        verbose=False,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
        )


class AgentService:

    async def generate_agent_response(self, user_input: str) -> str:
        response =  agent(user_input)
        if response is None:
            raise Exception("No response from agent")
        
        return response["output"]