# Import necessary modules
import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from train import init_pinecone_index
from RAG import init_rag
from utils import add_sidebar, format_sources

# Set up the Streamlit app
st.set_page_config(page_title="MadKudu: Support Rubook Chat 🧠", page_icon=":robot_face:")
st.title("🤖 MadKudu: Chat with our Notion Support Runbooks 🧠")

# Set up the sidebar
add_sidebar(st)

# initialize the variables
with st.spinner("Initializing..."):
        # get the index
        index_name = 'notion-db-chatbot'
        openai_api_key, vectordb = init_rag(index_name)
st.success("Ready to go! Please write your question in the chat box below", icon="✅")

# initialize the LLM
llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

template = """You are a support agent who knows the knowledge base inside-out.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Tell the user they might need to create a runbook to address this specific question.
Keep the answer concise.
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",input_key='question', output_key='answer', return_messages= True
    )

# create the function that retrieves source information from the retriever
def query_llm_with_source(retriever, query):
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        memory=st.session_state.memory,
        retriever=retriever
    )
    results = qa_chain({'question': query
                        ,'chat_history': st.session_state.messages
                        ,'rag_prompt': rag_prompt_custom
                        })
    st.session_state.messages.append((query, results['answer'] + "\n\n" + format_sources(results['sources'])))
    return results

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# st.session_state.retriever = retriever

if "messages" not in st.session_state:
    st.session_state.messages = []    
#
for message in st.session_state.messages:
    st.chat_message('human').write(message[0])
    st.chat_message('ai').write(message[1])    
#
if query := st.chat_input():
    st.chat_message("human").write(query)
    results = query_llm_with_source(retriever, query)
    answer = results['answer']
    sources = format_sources(results['sources'])
    st.chat_message("ai").write(answer + "\n\n" + sources)

