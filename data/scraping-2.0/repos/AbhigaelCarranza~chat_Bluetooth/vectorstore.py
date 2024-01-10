from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

#Crear Class para vectorstore
class VectorStore:  
    
    def get_vectorStore():
        load_dotenv()
        file="/Users/apple55/Github/Langchain/chat_Bluetooth/Files/bluetooth-act.pdf"
        pdf=PyPDFLoader(file)
        chunks=pdf.load_and_split()
        
        store_name=file[53:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store=FAISS.from_documents(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            # st.write("Vector store created")
        return vector_store
    
    