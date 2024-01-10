import os
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st

os.environ["OPENAI_API_KEY"] = 'personal-key'

default_doc_name = 'doc.pdf'

def process_doc(
    path: str 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
    is_local: bool = False,
    question: str = 'Cuáles son los autores del pdf?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)
        
    doc = loader.load_document()
    
    print(doc[-1])

    doc = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(doc, embedding = embeddings)
 
    qa = RetrievalQA.from_documents(llm= OpenAI(), chain_type='stuff', retriever=db.as_retriver())


    embeddings_model = OpenAIEmbeddings(OPENAI_API_KEY)
    
    st.write(qa.run(question))

def client():
    st.title('Manage LLM with LangChain')
    uploader = st.file_uploader('Upload a PDF file', type='pdf')
    
    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF saved!!')
        
    question = st.text_input('Generar un resumen de 20 palabras sobre el pdf',
                             placeholder= 'Give response about your PDF',
                             disable= not uploader)
    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('oading default PDF')
            process_doc()

if __name__ == '__main__':
    client()