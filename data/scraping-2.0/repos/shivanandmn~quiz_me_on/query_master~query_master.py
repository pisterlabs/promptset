from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from get_parms import openai_key
import openai
openai.api_key = openai_key()

def load_file(file:Path):
    if "pdf" in file:
        loader = PyPDFLoader(file)
        documents = loader.load()
    else:
        raise NotImplementedError
    return documents

def get_retriever(documents, embd_type:str="openai", search_type="similarity", search_kwargs=None):
    # define embedding
    if embd_type == "openai":
        embeddings = OpenAIEmbeddings()
    else:
        raise NotImplementedError
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(documents, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever

def text_splits(documents):
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs

def load_query_db(file:Path, chain_type, k):
    # load documents
    documents = load_file(file)
    documents = text_splits(documents)
    retriever = get_retriever(documents, embd_type="openai", search_type="similarity", search_kwargs={"k":k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # create a chatbot chain. Memory is managed externally.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), 
        chain_type=chain_type, 
        retriever=retriever,
        memory=memory
    )
    return qa_chain

if __name__ == "__main__":
    file = "query_master/docs/MachineLearning-Lecture01.pdf"
    qa_chain = load_query_db(file, chain_type="stuff", k=4)
    questions = "who are TA in this lecture?"
    results = qa_chain({"question":questions})
    print(results["answer"])



