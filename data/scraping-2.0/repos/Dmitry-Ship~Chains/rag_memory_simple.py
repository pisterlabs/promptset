from utils.rag import retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationKGMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory, ConversationBufferMemory
from utils.llm import llm

rag_chain_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

while True:
    query = input("\nUser: ")
    rag_chain_with_memory.invoke({
        "question": query, 
    })
