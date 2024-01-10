# how run it?
# install pip in your system by following the link(https://www.geeksforgeeks.org/how-to-install-pip-on-windows/)
# now run the command (pip install time, pip install langchain and pip install gradio) in command prompt
# ensure that llama-2-7b-chat.ggmlv3.q4_0.bin is present in the same directory
# then execute the python file using the command(python script.py )
# follow the link that appears on command prompt

# what is it?
# A customised chatbot for your personal use
# A chatbot that lets you talk with your own document
# Libraries of langchain that will be used are 
# 1. Document loader for retrieval
# 2. RecursiveCharacterTextSplitter
# 3. HuggingFaceEmbeddings
# 4. FAISS(A knowledge base offered by facebook)
# 5. Chains
from langchain.document_loaders import CSVLoader, PyPDFLoader,JSONLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

import gradio as gr
import time
file=""
data=[]
DB_FAISS_PATH=""
qa=""
chat_history = []
# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
text_chunks=[]

# upload file function
def upload_file(files):
    file_paths = [file.name for file in files]
    load(file_paths)
    return file_paths

# processing of the data
def load(files):
    for file in files: 
        DB_FAISS_PATH = "vectorstore/db_faiss"+file[ file.rindex('\\') : file.rindex('.') ]
        if(file.endswith('.csv')):
            loader = CSVLoader(file_path=file, encoding="utf-8", csv_args={'delimiter': ','})
        elif(file.endswith('.pdf')):
            loader = PyPDFLoader(file)
        elif(file.endswith('.json')):
            loader = JSONLoader(file_path=file)
        elif(file.endswith('.txt')):
            loader= TextLoader(file_path=file)

    global data, text_chunks

    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    # At a high level, text splitters work as following:
    # Split the text up into small, semantically meaningful chunks (often sentences).
    # Start combining these small chunks into a larger chunk until you reach a certain size (as measured by some function).
    # Once you reach that size, make that chunk its own piece of text and then start creating a new chunk of text with some overlap (to keep context between chunks).
    text_chunks = text_splitter.split_documents(data)# creates a list of chunks

    # Download Sentence Transformers Embedding From Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local(DB_FAISS_PATH)# saving the embedding locally on the system

    print(data[0])

def user(user_message, history):
            return "", history + [[user_message, None]]

def bot(history):

    # COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
    docsearch = FAISS.from_documents(text_chunks, embeddings)

    docsearch.save_local(DB_FAISS_PATH)
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.1)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())
    bot_message = qa({"question":history[-1][0],'chat_history':chat_history})['answer']
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [ msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()



