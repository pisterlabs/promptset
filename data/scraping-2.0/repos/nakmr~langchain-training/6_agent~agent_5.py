from json import tool
from tabnanny import verbose
from unittest import result
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

tools = []

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
    top_k_results=2
)

tools.append(
    create_retriever_tool(
        name="WikipediaRetriever",
        description="Retrieve the Wikipedia page of the specified keyword.",
        retriever=retriever
    )
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

result = agent.run("バーボンウイスキーの歴史を調べて、概要を日本語で、result.txtというファイルに保存してください。")
print(f"実行結果 1回目: {result}")

result = agent.run("以前の作業をもう一度実行してください。")
print(f"実行結果 2回目: {result}")