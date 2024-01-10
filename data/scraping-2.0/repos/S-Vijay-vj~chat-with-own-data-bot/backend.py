# importing libraries
from PyPDF2 import PdfReader
from langchain import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# creating function to read the contents of the PDF
def pdf_reader(pdf_doc):
    text = ''    
    # for pdf in pdf_doc:
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# creating function to read the contents of the CSV and create chunks
def csv_reader(csv_doc):

    loader = CSVLoader(csv_doc)
    data = loader.load()
    return data


# creating function to create chunks 
def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=100,
        length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

# creating function to create and store embeddings
def get_vectorstore(chunks,user_key):
    embeddings = OpenAIEmbeddings(openai_api_key=user_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Creating conversation chain
def get_conversation_chain(vectorstore,user_key):
    llm = ChatOpenAI(temperature=0.0,openai_api_key=user_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory =ConversationBufferMemory(
        memory_key='chat_history',  
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
