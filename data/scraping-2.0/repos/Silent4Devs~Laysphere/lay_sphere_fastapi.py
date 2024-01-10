from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class FileUpload(BaseModel):
    chunk_size: int
    k: int
    file: UploadFile

class Question(BaseModel):
    q: str

vector_store = None
chat_history = []

### CARGAR DOCUMENTOS ###

def load_document(file):
    name, extension = os.path.splitext(file.filename)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file.filename}')
        loader = PyPDFLoader(file.file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file.filename}')
        loader = Docx2txtLoader(file.file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file.file)
    else:
        print("Documento no soportado")
        return None

    data = loader.load()
    return data

### CARGAR CHUNKS ###

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

### CARGAR EMBEDDINGS ###

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

### SPEAKING ###

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='text-embedding-ada-002', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(q)
    return answer

### calcular costos del embedding ###

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file_upload: FileUpload):
    global vector_store, chat_history
    chunk_size = file_upload.chunk_size
    k = file_upload.k

    bytes_data = await file_upload.file.read()
    file_name = os.path.join('./', file_upload.file.filename)
    with open(file_name, 'wb') as f:
        f.write(bytes_data)

    data = load_document(file_upload.file)
    chunks = chunk_data(data, chunk_size=chunk_size)

    tokens, embedding_cost = calculate_embedding_cost(chunks)

    vector_store = create_embeddings(chunks)
    chat_history = []

    return templates.TemplateResponse(
        "upload_complete.html",
        {"request": None, "chunk_size": chunk_size, "embedding_cost": embedding_cost},
    )

@app.post("/ask", response_model=dict)
def ask_question(question: Question):
    global vector_store, chat_history
    q = question.q
    if vector_store is not None:
        result, chat_history = ask_with_memory(vector_store, q, chat_history)
        answer = result['answer']
        return {"answer": answer}
    else:
        return {"answer": "Please upload a file and chunk it first."}

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0.1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

@app.get("/", response_class=HTMLResponse)
def read_root():
    return templates.TemplateResponse("index.html", {"request": None})
