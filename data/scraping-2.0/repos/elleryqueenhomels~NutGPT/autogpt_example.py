import faiss
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS

from autogpt.agent import AutoGPT


# Define tools for AI agent to use
web_search = DuckDuckGoSearchRun()
tools = [
    web_search,
    WriteFileTool(),
    ReadFileTool(),
]

# Define embedding model for VectorDB
embeddings_model = OpenAIEmbeddings()

# Memory
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Create a LLM client
llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

# Create an agent
agent = AutoGPT.from_llm_and_tools(
    ai_name="ChatGPT",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(),
    verbose=True,
)
# Set verbose to be true
agent.chain.verbose = True

agent.run(["write a weather report for Dongguan city in China today"])
