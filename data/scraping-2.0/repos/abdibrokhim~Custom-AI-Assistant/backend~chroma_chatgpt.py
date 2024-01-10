import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv


def generate_prompt(query, file_path="backend/files/sample.pdf"):
    
    load_dotenv()

    OPENAI_API_KEY=""
                    
    try:

        print('query:', query)
        print('file_path:',file_path)

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma.from_documents(texts, embeddings)

        qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=vectordb)

        prompt = qa.run(query)

        if prompt == "":
            return ""
        
        print('prompt:', prompt)
        
        return prompt.strip()

    except Exception as e:
        print(e)
        return ""