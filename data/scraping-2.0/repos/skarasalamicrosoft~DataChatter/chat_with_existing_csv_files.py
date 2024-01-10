import os
import openai
import sys
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

sys.path.append('../..')

import os
import openai
import sys
import streamlit as st
sys.path.append('../..')


api_key = "43a50c02313e404c9096bb7c69173d1d"
api_base = "https://cgaddam-openai.openai.azure.com/"
openai.api_version = '2023-05-15' # may change in the future
openai.api_type = 'azure'

current_directory = os.getcwd()
print("Current Directory:", current_directory)


from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    openai_api_key=api_key,
    openai_api_base=api_base,
    openai_api_version='2023-05-15',
    engine='cgaddamembeddings'
)

from langchain.vectorstores import Chroma
persist_directory = 'docs/people'

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key,
    openai_api_base=api_base,
    engine='cgaddamllm'
)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

def main():
    # Set the title and subtitle of the app
    st.title('🦜🔗 Chat With CSV')
    st.subheader('Choose your CSV file, ask questions, and receive answers directly from the file.')

    
    # selected_folder = ""
    
    doc_directory = f"{current_directory}/docs"
    files = [f for f in os.listdir(doc_directory) if os.path.isdir(os.path.join(doc_directory, f))]
    file = st.selectbox("Choose a file", files)
    file_path = f"{current_directory}/{file}.csv"
    df = pd.read_csv(file_path)
    st.write(df)
    
    question = st.text_input("Type your question")

    persist_directory = f'docs/{file}'

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    if st.button("Submit Query", type="primary"):
        
        response = qa({"question": question})
        
        st.write(response['answer'])
        

if __name__ == '__main__':
    main()
