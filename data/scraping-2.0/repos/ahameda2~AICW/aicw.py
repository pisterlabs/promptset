import streamlit as st
from huggingface_hub import notebook_login
from PyPDF2 import PdfReader
import io
import os
import fs
from langchain.embeddings import HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, HNSWLib
from langchain.llms import CTransformers, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain.schema.output_parser import StrOutputParser
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Setting up the Streamlit page
st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")

# Define a class to hold the text and metadata with the expected attributes
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# Define the function to read and extract text from a PDF byte stream
def read_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ""  # Adding a fallback for pages with no text
    return text

# Sidebar for Hugging Face Login and PDF Upload
with st.sidebar:
    st.subheader("Hugging Face Login")
    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    submit_button = st.button("Login")

    if submit_button:
        try:
            notebook_login(hf_token)
            st.success("Connected successfully to Hugging Face Hub.")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

    st.subheader("Your Documents")
    uploaded_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')

# Main Page Interface
st.header("Chat with Multiple PDFs 📚")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # For Apple Silicon GPUs
    else:
        return "cpu"

DEVICE = get_device()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Processing PDFs
if uploaded_files:
    documents = []
    for file in uploaded_files:
        file_stream = io.BytesIO(file.getbuffer())
        document_text = read_pdf(file_stream)
        documents.append(Document(document_text))

    st.session_state.documents_processed = True
    st.success("PDFs processed successfully!")

    # Combine all texts and split into chunks
    combined_text = " ".join([doc.page_content for doc in documents])
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_text = text_splitter.split_documents(documents)
    st.write(f"Number of text chunks: {len(split_text)}")

    # Creating database of embeddings
    db = Chroma.from_documents(split_text, embeddings, persist_directory="db")
    st.success("Embeddings processed and database created.")

    # Initialize the model and tokenizer for conversational AI
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, torch_dtype='auto')
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, torch_dtype='auto').to(DEVICE).half()
    max_seq_length = 128

    # Define a function to format chat history
    def formatChatHistory(human, ai, previousChatHistory=None):
        newInteraction = f"Human: {human}\nAI: {ai}"
        if not previousChatHistory:
            return newInteraction
        return f"{previousChatHistory}\n\n{newInteraction}"

    # Define a prompt template for generating an answer
    questionPrompt = PromptTemplate.fromTemplate(
        """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ----------------
        CONTEXT: {context}
        ----------------
        CHAT HISTORY: {chatHistory}
        ----------------
        QUESTION: {question}
        ----------------
        Helpful Answer:""")

    # Define the conversational chain
    chain = RunnableSequence.from([
        # Your existing code to process the question, chat history, and context
        questionPrompt,
        model,
        StringOutputParser(),
    ])

    # Chat interface
    st.subheader("Chat with your PDFs")
    user_query = st.text_input("Ask a question about your documents:", key="user_query")
    if st.button("Submit"):
        if user_query:
            context = combined_text  # Use combined text from documents
            chat_history = formatChatHistory("User", user_query)  # Format chat history
            result = chain.invoke({
                'question': user_query,
                'chatHistory': chat_history,
                'context': context
            })
            st.write('Answer:', result['resultOne'])
        else:
            st.warning("Please enter a question.")

