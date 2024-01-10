from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
vector_store = None
conversation_chain = None 

pdfname=None
pdfsize=None
scripttext=None
import fitz  # PyMuPDF

def get_pdf_text(pdf_docs):
    global pdfname, pdfsize, scripttext
    text = ""
    for pdf in pdf_docs:
        print("Uploaded PDF name:", pdf.name)
        pdfname=pdf.name
        print("Uploaded PDF size:", pdf.size)
        pdfsize=pdf.size
        doc = fitz.open("pdf", pdf.read())
        for page in doc:
            text += page.get_text()
            
            
    print("text", text)
    scripttext=text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    # print(chunks)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def process_uploaded_pdfs(uploaded_pdfs):
    global vector_store, conversation_chain
    global pdfname, pdfsize, scripttext
    raw_text = get_pdf_text(uploaded_pdfs)
    # print("Extracted PDF text:", raw_text)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)
    return({"pdfname":pdfname,"pdfsize": pdfsize,"scripttext": scripttext})

def handle_user_question(user_question):
    global conversation_chain
    if not conversation_chain:
        return {"status": "error", "message": "Please upload and process PDFs first."}
    response = conversation_chain({'question': user_question})
    return {
        "status": "success",
        "user_question": user_question,
        "bot_response": response['chat_history'][-1].content if response['chat_history'] else "Sorry, I couldn't understand that."
    }


    
    