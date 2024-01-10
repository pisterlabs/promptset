import man_find
import os 
import pinecone 
from tqdm import tqdm
import time 
import threading
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI

def loading_animation(event): 
    symbols = ['|', '/', '-', '\\']
    current = 0

    while not event.is_set():
          print(symbols[current], end='\r')
          time.sleep(0.1)
          current = (current + 1) % len(symbols)
    print("Done")

load_dotenv(find_dotenv(), override=True)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)

# make the man pages into chunks of 1024 characters
def chunk_data(data, chunk_size=1024):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    print("Chunking documents")
    chunks = [chunk for chunk in tqdm(text_splitter.split_documents(data), desc="Chunking documents")]
    return chunks

# upload the chunks to pinecone 
def insert_or_fetch_embedding(index_name, chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))

    print("Creating index...")
    #start loading animation
    stop_loading_event = threading.Event()
    loading_event = threading.Thread(target=loading_animation, args=(stop_loading_event,))
    loading_event.start()

    try: 
        if index_name in pinecone.list_indexes():
            # index already exists, fetch it 
            vector_store = Pinecone.from_existing_index(index_name, embeddings)
        else: 
            # create new index 
            pinecone.create_index(name=index_name, dimension=1536, metric='cosine')

            vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    finally: 
         #stop loading animation 
         stop_loading_event.set()
         loading_event.join()
    return vector_store

# delete index in pinecone (free tier needs this)
def delete_pinecone_index(index_name='all'):
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
    print("Deleting index...")
    stop_loading_event = threading.Event()
    loading_event = threading.Thread(target=loading_animation, args=(stop_loading_event,))
    loading_event.start()
    try:
        if index_name == 'all':
            indexes = pinecone.list_indexes()
            for index in indexes:
                pinecone.delete_index(index)
        else:
            pinecone.delete_index(index_name)

    finally: 
        stop_loading_event.set()
        loading_event.join()

def asked_question(vector_store, message):
    from langchain.chains import RetrievalQA
    print("Coming up with an answer...")
    stop_loading_event = threading.Event()
    loading_event = threading.Thread(target=loading_animation, args=(stop_loading_event,))
    loading_event.start() 
    try:
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        answer = chain.run(message)
    finally: 
        stop_loading_event.set()
        loading_event.join()
    return answer

def command_message(command, message):
    data = man_find.find_documentation(command)

    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("/tmp/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)

    data = loader.load()

    if data:
            chunks = chunk_data(data)
            index_name = 'manai'

            vector_store = insert_or_fetch_embedding(index_name, chunks)

            answer = asked_question(vector_store, message)

            delete_pinecone_index(index_name) # keep free teir happy
            
            return answer
    else:
         message = "Sorry, I couldn't find any documentation for the command. I can give you a description of what the command does. Would you like that?"
         return message


def manai_command(command):
    # do the chat model stuff
    message = f"Can you give me a description of the {command} command As well as a few examples describing how to use it as a beginner, intermediate, and expert commands?"
    response = command_message(command, message)
    
    return response

def manai_message(message):
    # this is where the user can just send a normal asking message to the chat
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
    response = llm(message)
    response_content = response.content if hasattr(response, 'content') else ''
    return response_content 


def manai_command_message(command, message):
    # this is where the user can ask a specific question about a command
    response = command_message(command, message)
    return response 