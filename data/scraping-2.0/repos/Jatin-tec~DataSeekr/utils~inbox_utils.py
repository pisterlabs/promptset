import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain import HuggingFaceHub
from langchain.prompts.chat import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from utils.htmlTemplate import bot_template, user_template

def get_inbox_conversation_chain(vectorstore):
    momory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, verbose=True, input_key="question", output_key="answer")
    # llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.7, "max_length": 10000})
    llm = OpenAI(temperature=0.8, max_tokens=2000)

    system_template = """Use following pieces of context to answer the users question. 
    Following contexts are summarised texts from users inbox. If you don't find the answer, ask again with more details.
    ----------------
    {context}"""

    # Create the chat prompt templates
    messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)
    retriver = vectorstore.as_retriever(search_kwargs={"k": 10})

    conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriver, 
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    max_tokens_limit=4000,
    memory=momory,
    verbose=True
    )
    return conversation_chain


def inbox_user_input(user_query):
    response =  st.session_state.conversation({"question": user_query})

    st.session_state.chat_history = response['chat_history']
    answer = response['answer']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
