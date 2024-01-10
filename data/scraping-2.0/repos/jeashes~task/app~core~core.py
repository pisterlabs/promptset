from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import OpenAI

from utils.helper_func import verify_maxtokens

def set_question(question: str) -> str:

    verify_maxtokens(question)

    load_dotenv()
    embeddings = OpenAIEmbeddings()

    loader = PyPDFLoader('documents/parser-paper.pdf')
    document = loader.load_and_split()

    text_splitter = CharacterTextSplitter(chunk_size=2058)
    texts = text_splitter.split_documents(document)

    vecstore = FAISS.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type='stuff', 
        retriever=vecstore.as_retriever()
    )

    answer = qa.run(question)
    if "I don't know" in answer:
        answer = "I don`t know please contact with support by email support@nifty-bridge.com"
    
    return answer
    
