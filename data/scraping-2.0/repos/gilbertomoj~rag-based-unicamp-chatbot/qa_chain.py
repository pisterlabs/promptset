import os
from dotenv import load_dotenv
from extractor import read_pdf_info
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
path = "output/data.txt"


def setup_qa_chain():
    if not os.path.exists(path):
        read_pdf_info()
    loader = TextLoader("output/data.txt", "utf-8")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vector_store = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    llm = ChatOpenAI(model='gpt-3.5-turbo')

    retriever = vector_store.as_retriever()

    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    return qa_chain
