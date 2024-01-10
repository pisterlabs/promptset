import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import openai
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader

def comp_process(apikey, pdfs, question):
    
    #Initialize our Language Model
    os.environ["OPENAI_API_KEY"] = apikey
    llm = openai(temperature=0, openai_api_key=apikey)
    
    text = ""
    
    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text=text)
    
    embedding = OpenAIEmbeddings(openai_api_key=apikey)
    docsearch = Chroma.from_texts(chunks, embeddings=embedding).as_retriever     
    
    if question:
        docs = docsearch.get_relevant_documents(question)  
        read_chain = load_qa_chain(llm=llm) 
        answer = read_chain.run(input_documents=docs, question=question)
        
    return (answer) 
        
    
    