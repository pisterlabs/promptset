from langchain.agents import Tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.utilities import SerpAPIWrapper
from langchain.memory.chat_message_histories import FileChatMessageHistory

search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True


# agent = AutoGPT.from_llm_and_tools(
#     ai_name="Tom",
#     ai_role="Assistant",
#     tools=tools,
#     llm=ChatOpenAI(temperature=0),
#     memory=vectorstore.as_retriever(),
#     chat_history_memory=FileChatMessageHistory("chat_history.txt"),
# )
