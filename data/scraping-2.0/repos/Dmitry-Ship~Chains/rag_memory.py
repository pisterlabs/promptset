from utils.rag import create_rag_chain_with_memory, retriever
from utils.llm import llm
from langchain_core.messages import HumanMessage, AIMessage

rag_chain_with_memory = create_rag_chain_with_memory(llm, retriever)
chat_history = []

while True:
    query = input("\nUser: ")
    ai_msg = rag_chain_with_memory.invoke({
        "question": query, 
        "chat_history": chat_history
    })

    chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg)])
