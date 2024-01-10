from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def extract_data_from_pdfs(docs):
    """
    Extracts text from the pdf and return in a string
    """
    text_list = []
    for doc in docs:
        reader = PdfReader(doc)

        # iterating through pages and getting text
        for page in reader.pages:
            text_list.append(page.extract_text())

    return " ".join(text_list)


def create_text_chunks(data):
    """
    Chunks the string data and returns the chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(data)
    return text_chunks


def create_vectorstore(text_chunks):
    """
    Function to develop vectorstore from chunks of data retrieved form the pdfs
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    """
    Creates a conversation chain to store history of conversation between user and llm
    """
    llm = ChatOpenAI(streaming=True)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

