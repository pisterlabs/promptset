from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
# from langchain.document_loaders import TextLoader
from langchain.document_loaders import WhatsAppChatLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import pinecone
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


load_dotenv()


if __name__ == '__main__':
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    #docsearch = Pinecone.from_documents(texts, embeddings, index_name='whatsapp-bruna-embeddings-index')
    #docsearch = Pinecone.from_existing_index(embedding=embeddings, index_name='whatsapp-bruna-embeddings-index')
    # local machine
#   #vectorstore = FAISS.from_documents(texts, embeddings)
    #vectorstore.save_local("faiss_index_lf")

    new_vectorstore = FAISS.load_local(r"C:\trabalho_upwork\langchan_course\luiza_ferraz\faiss_index_lf", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())

    res = qa.run("Quanto custa o produto que tem o codigo igual a 91713?")
    print(res)