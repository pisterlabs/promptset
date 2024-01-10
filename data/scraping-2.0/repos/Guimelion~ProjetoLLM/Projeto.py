import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA

st.markdown('''# Pergunte ao seu livro''')

arquivo = st.file_uploader('Suba seu arquivo aqui (apenas .TXT)',type=['txt'])

if arquivo is not None:
    #loader = TextLoader(arquivo)
    #documents = loader.load()
    bytes_data = arquivo.read()
   
    
    with open("ARQUIVO.txt", "w",encoding="utf-8") as file:
        file.write(bytes_data.decode())

    #Carregando os dados do texto
    loader = TextLoader('ARQUIVO.txt',encoding = 'UTF-8')
    documents = loader.load()

    #Quebrando o texto em partes menores
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    #Armazenando em um vectorDb para fazer busca semantica
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts,embeddings)

    #Exibindo as partes do texto mais semelhantes com o que você procura
    query = st.text_input('Digite o que procura:')
    texto = db.similarity_search(query)
    st.write(texto)

    #Este trecho foi uma tentativa de trazer uma resposta usando llm

    #llm = HuggingFaceHub(repo_id='01-ai/Yi-34B',model_kwargs = {'temperature':0,'max_length':512})
    #chain = load_qa_chain(llm,chain_type='stuff')
    #texto = db.similarity_search(query)
    #st.write(chain.run(input_document=texto,question=query)))
else:
    st.error('Sem arquivo')

    

    