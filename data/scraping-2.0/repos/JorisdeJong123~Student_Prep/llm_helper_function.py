from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import SeleniumURLLoader, PyPDFLoader
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Function to extract text from PDF for question generation
def extract_text_from_pdf_for_q_gen(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    docs = split_text_q_gen(text)
    return docs

# Function to extract text from PDF for question answering
def extract_text_from_pdf_for_q_answer(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    docs = split_text_q_answer(text)
    return docs

# Text splitter when the input text is just text, not documents.
# Used for question generation
def split_text_q_gen(data):
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(data)
    docs = [Document(page_content=t) for t in texts]
    return docs

# Text splitter when the input text is just text, not documents.
# Used for question answering, vector database
def split_text_q_answer(data):
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_text(data)
    docs = [Document(page_content=t) for t in texts]
    return docs

# Function for splitting texts into documents
def split_text_docs_vector(data):
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

# Function to create a LLM model
def create_LLM(openai_api_key, temperature, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key,temperature=temperature, model=model_name)
    return llm

def load_data_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()
    return data

# Function to create a single Tweet
def create_questions(docs, llm):
    question_chain = load_summarize_chain(llm, chain_type="refine", verbose=True, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
    questions = question_chain.run(docs)
    return questions

def create_vectordatabase(docs, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db