import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
# from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

from langchain.chat_models import BedrockChat

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    
    # Load OpenAI chat model
    # llm = ChatOpenAI(temperature=0, max_tokens=2400, model="gpt-3.5-turbo-16k")

    llm = BedrockChat(
        model_id="anthropic.claude-v2",
        credentials_profile_name="default",
        region_name="us-east-1",
        model_kwargs={"temperature": 0, "max_tokens_to_sample": 2048}
    ) 

    
    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                  retriever=retriever, 
                                                  memory=memory, 
                                                  get_chat_history=lambda h : h,
                                                  verbose=True)

    # System prompt
    template = """
    You are the world's best assistant in academic information systems design. 

    You will be asked questions, and you will answer them using the Knowledgebase first, and only then try to find out the answer elsewhere. Your answers are detailed yet concise. Imagine you're are a student getting the Masters degree in Information systems design, and this is your exam.

    You don't make up anything. If you're not sure, you answer "I don't know 🤷‍♂️"

    Your language is Ukrainian, you answer in Ukrainian but you give the name of the terms in English too.

    Your answer should be not less than 3 sentences. Structure the answer using markdown into sections.

    модель ВВС (Взаємодії відкритих систем) == Open Systems Interconnection (OSI) model

    === Extracted parts ===
    {context}

    === Question === 
    {question}
    
    === Helpful Answer ==="""

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain