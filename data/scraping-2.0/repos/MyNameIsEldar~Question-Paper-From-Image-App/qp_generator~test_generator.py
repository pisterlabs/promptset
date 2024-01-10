import app
import streamlit as st
import qp_generator.templates as templates
from qp_generator.languages_data import languages
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate
)
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough

@st.cache_data
def main(qp_type, text, tasks, answers, lang):
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5, model_name='gpt-4')

    if qp_type == languages[lang]['test']:
        temp_part = templates.test
        temp_part = temp_part.format(type=qp_type, num_of_q=tasks, num_of_a=answers)
    if qp_type == languages[lang]['open_questions']:
        temp_part = templates.open_questions
        temp_part = temp_part.format(num_of_q=tasks)

    template = ("""
Ты полезный помошник-ассистент, который помогает пользователю составлять {type}.""" 
+ temp_part) + "Текст: {text}"
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    texts = text_splitter.split_text(text)

    db = Chroma.from_texts(texts, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = {'text': retriever, 'type': RunnablePassthrough()} | prompt | llm 

    result = rag_chain.invoke(qp_type).content
    
    return result
