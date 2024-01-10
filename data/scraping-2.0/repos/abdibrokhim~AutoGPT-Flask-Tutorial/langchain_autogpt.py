# We’ll set up an AutoGPT with a search tool, and write-file tool, and a read-file tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool


# The memory here is used for the agents intermediate steps
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

# Initialize everything! We will use ChatOpenAI model
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

# Initialize the vectorstore as empty
import faiss


search = SerpAPIWrapper(serpapi_api_key="4bce9706bf3ec449df8b03a228afbdae0ce5da06649b0d458bb223327efbf10d")
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key="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")

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(openai_api_key="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", temperature=0),
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True


agent.run(["generate a couple names for my dog."]) 
