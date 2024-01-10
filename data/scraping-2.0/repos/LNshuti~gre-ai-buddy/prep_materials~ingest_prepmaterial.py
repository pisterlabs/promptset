import streamlit as st
from llama_hub.file.base import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory

def load_documents():
    loader = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    return documents

def index_documents(documents):
    index = GPTVectorStoreIndex.from_documents(documents)
    return index

def initialize():
    documents = load_documents()
    index = index_documents(documents)
    tools = [
        Tool(
            name="Local Directory Index",
            func=lambda q: index.query(q),
            description=f"Useful when you want answer questions about the files in your local directory.",
        ),
    ]
    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_chain = initialize_agent(
        tools, llm, agent="zero-shot-react-description", memory=memory
    )

    return agent_chain

def app():
    st.title('Local File Query Tool')
    
    agent_chain = initialize()

    query = st.text_input("Enter your question about the files:")
    if query:
        output = agent_chain.run(input=query)
        st.write(output)

if __name__ == '__main__':
    app()
