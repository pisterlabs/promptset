from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import langchain
import requests
import io
import docx
import json

langchain.verbose = False
def get_text_from_files(file_list):
    text = ""
    for file in file_list:
        ext_name = file.split(".")[-1].lower()
        response = requests.get(file)
        if ext_name == "pdf":   
            ##load pdf from url          
            pdf_reader = PdfReader(io.BytesIO(response.content))  
            # pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif ext_name == "txt":
            #load txt from url
            text = response.content.decode("utf-8")
        elif ext_name == "docx":
            #load docx from url
            docx_file = io.BytesIO(response.content)
            doc = docx.Document(docx_file)
            for para in doc.paragraphs:
                text += para.text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

##get keys from .env file
import os
# from dotenv import load_dotenv
# load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     temperature=0.25)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def process(text_info,files_or_urls,questions):
    conversation=None

    if text_info:   
        raw_text = str(text_info)
    else:
        raw_text = ""

    ## allow if only text is input
    if files_or_urls:
        raw_text = raw_text+ "\n" + str(get_text_from_files(files_or_urls))
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)
    for user_question in questions:
        response = conversation({'question': user_question})
    chat_history = response['chat_history']
    data={}
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            key=message.content
        else:
            data[key]=message.content
    return data

    
  


def lambda_handler(event, context):
    # TODO implement
    result=process(text_info=event['text_info'],files_or_urls=event['files_or_urls'],questions=event['questions'])
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }

