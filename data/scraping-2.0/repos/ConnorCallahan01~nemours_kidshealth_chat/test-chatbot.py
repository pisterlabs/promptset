import utils
import streamlit as st
from streaming import StreamHandler

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferWindowMemory
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.agents import AgentType, initialize_agent
import requests
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.prompts import PromptTemplate
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from chains import VectorDBChain

st.title("💬 Nemours KidsHealth Chatbot")
st.subheader("Trained on the section: General Health --> Pains, Aches, & Injuries")
st.write("Chat with KidsHealth data to get more custom information regarding your child's health!")
st.write("Ask questions like: 'My child fell of their bike a week ago and is still complaining of pain, what should I do?', 'My child has a fever and is complaining of a headache, what should I do?', 'My child has a sprained ankle, what should I do?'")
if st.button("To reset chat, press this and refresh the page."):
    # Clear values from *all* all in-memory and on-disk data caches:
    # i.e. clear values from both square and cube
    st.cache_data.clear()
class ContextChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = st.secrets["model"]
    
    @st.cache_resource
    def setup_chain(_self):
        # loader_pinecone = PyPDFDirectoryLoader("./Aches_Pains_&_Injuries/")
        # pages = loader_pinecone.load()
        # model_name = 'text-embedding-ada-002'
        # embeddings_pinecone = OpenAIEmbeddings(model=model_name, openai_api_key=st.secrets["open_ai_api"])
        # pinecone.init(api_key=st.secrets["pinecone_api"],environment=st.secrets["environment"])

        # index_name = "buzzindex"
        # index = pinecone.Index(index_name)
        # text_field = "text"

        llm = ChatOpenAI(
            temperature=0.8,
            model_name=st.secrets["model"],
            openai_api_key=st.secrets["open_ai_api"]
        )
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        vdb = VectorDBChain(
            index_name="buzzindex",
            environment=st.secrets["environment"],
            pinecone_api_key=st.secrets["pinecone_api"]
        )

        vdb_tool = Tool(
            name=vdb.name,
            func=vdb.query,
            description="This tool allows you to get references to the query from the documents. Use the returned text as knowledge for how to answer the query."
        )
        system_message = """You are a medical assistant answering questions that parents have about their childrens' health. 
                            Be conversational, ask follow-up questions, and use tools and the chat history to answer the user's question"""
        
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=[vdb_tool],
            llm=llm,
            verbose=True,
            agent_kwargs={"system_message": system_message},
            memory=conversational_memory,
            handle_parsing_errors=True,
            
        )
        return agent
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.spinner("Thinking..."):
                st_cb = StreamHandler(st.empty())
                response = chain(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
            utils.display_msg(st.session_state.messages[-1]['content']["output"], 'assistant')
if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
