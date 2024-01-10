import streamlit as st
import os
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain 
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pandas as pd

os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(model_name='gpt-3.5-turbo')

df = pd.read_csv("DisneylandReviews.csv", encoding="latin-1")    
if pdf is not None:
        with st.status("Extrayendo texto...", expanded=True) as status:
         
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text+= page.extract_text()

            #langchain_textspliter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )

            st.write('Creando chunks...')
            chunks = text_splitter.split_text(text=text)

        
        
            store_name = pdf.name[:-4]
        
            st.write('Buscando Vectorstore..')
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl","rb") as f:
                    vectorstore = pickle.load(f)
                st.write("Vector store existente. Cargando...")
            else:
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

                with open(f"{store_name}.pkl","wb") as f:
                    pickle.dump(vectorstore,f)
            
                st.write("Guardando embeddings en memoria...")

            status.update(label="Todo listo para hacer preguntas!", state="complete", expanded=False)

        query = st.text_input("Preguntale a tu PDF. Pica enter luego de escribir tu pregunta","EG: De que habla este documento")
        if query !="EG: De que habla este documento":
            with st.spinner('Trabajando en eso...'):
                docs = vectorstore.similarity_search(query=query,k=3)
                chain = load_qa_chain(llm=llm, chain_type= "stuff", verbose=True)
                with get_openai_callback() as cb:
                    response = chain.run(input_documents = docs, question = query)
                    print(cb)

                st.success(f'''
                    RESPUESTA:
                    {response}
                ''', icon="✅")




