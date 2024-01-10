import os
import multiprocessing
import torch
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from langchain.llms import LlamaCpp
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from pdf_parser import pdf_to_ocr

load_dotenv()

# GPU Inference
cuda_available = 0
if (torch.cuda.is_available()):
    print ('Nvidia GPU detected!')
    os.environ['LLAMA_CPP_LIB'] = os.getenv('LLAMA_CPP_LIB', 'usr/local/lib/libllama.so')
    os.environ['LLAMA_CUBLAS'] = os.getenv('LLAMA_CUBLAS', 'on')
    cuda_available = 1
else:
    print ('Defaulting to CPU!')

# Model initialization
MODEL_PATH = os.getenv('MODEL_PATH')
if cuda_available:
    # GPU Layers = 25 acceptable for 4GB VRAM
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=25, max_tokens=2048, temperature=0, n_batch=512)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )
else:
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, n_threads=multiprocessing.cpu_count(), temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

# Chroma DB
persist_directory = os.getenv('PERSIST_DIRECTORY')
CHROMA_SETTINGS = Settings(persist_directory=persist_directory, anonymized_telemetry=False)
chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)

# Prompts
SUMMARY_PROMPT_TEMPLATE = """
### System: 
You are an AI assistant. You will be given a task. You must generate a detailed and long answer.

### User:
Summarize the following text.
{text}

### Response:
Sure, here is a summary of the text:
"""

GRAMMAR_PROMPT_TEMPLATE = """
### System: 
You are an AI assistant that follows instruction extremely well. Help as much as you can.

### User:
Read the following text, and rewrite all sentences after correcting all the writing mistakes:
{text}

### Response:
"""

CHAT_PROMPT_TEMPLATE = """
### System:
You are an AI assistant that helps people find information.

### User:
This is your previous chat history, where "Human:" is the user's query and "AI:" is your response to the query:
{chat_history}

This is the information provided to you:
{context}

Use only the conversation history (if there was previous conversations made) and the information to answer the following query.
{question}

### Response:
"""

summarize_prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
grammar_prompt = PromptTemplate(template=GRAMMAR_PROMPT_TEMPLATE, input_variables=["text"])
qa_prompt = PromptTemplate(template=CHAT_PROMPT_TEMPLATE, input_variables=["question", "chat_history", "context"])

def summarize_pdf(pdf_path):
    # Convert pdf to text
    text = pdf_to_ocr(pdf_path)
    # Split text into chunks
    text_splitter = TokenTextSplitter()
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    summary_chain = load_summarize_chain(llm, chain_type='stuff', prompt=summarize_prompt)
    # Run inference
    try:
        result = summary_chain.run(docs)
    except Exception as e:
        return e

    return result

def grammar_check(text):
    llm_chain = LLMChain(prompt=grammar_prompt, llm=llm)
    # Run inference
    try:
        result = llm_chain.run(text)
        return result
    
    except Exception as e:
        return e

def ingest_file(pdf_path):
    # Convert pdf to text
    try:
        text = pdf_to_ocr(pdf_path)
        text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=0)
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        db = Chroma.from_documents(docs, embeddings, 
                                   persist_directory=persist_directory, 
                                   client=chroma_client,
                                   client_settings=CHROMA_SETTINGS)
        print ('File has been ingested!')
        return "File has been uploaded!"
    except Exception as e:
        return e

def chat_qa(query, chat_history):
    # Use stored embeddings
    db = Chroma(persist_directory=persist_directory, 
                embedding_function=embeddings, 
                client_settings=CHROMA_SETTINGS, 
                client=chroma_client)
    print (db.get())
    # Initialize chat memory, uses chat state from frontend
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
    for index in range(0, len(chat_history), 2):
        prev_user_msg, prev_ai_msg = chat_history[index], chat_history[index+1]
        memory.chat_memory.add_user_message(prev_user_msg)
        memory.chat_memory.add_ai_message(prev_ai_msg)
    
    qa = load_qa_chain(llm=llm, memory=memory, prompt=qa_prompt, verbose=True) # Initialize QA
    docs = db.similarity_search(query, k=8) # Get relevant docs

    result = qa({
        "input_documents" : docs,
        "question" : query
    })
    print (result, flush=True)

    return result['output_text']

# Sample inference
if __name__ == "__main__":
    pdf_path = "./samples/tiny-attention.pdf"
    res = summarize_pdf(pdf_path)
    print (res)